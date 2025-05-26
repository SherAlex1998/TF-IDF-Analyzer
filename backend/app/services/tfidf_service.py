from __future__ import annotations
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from typing import List, Dict, Union, Any

class TFIDFService:
    def __init__(self, language: str = "english"):
        for pkg, path_part in [("punkt", "tokenizers/punkt"),
                               ("stopwords", "corpora/stopwords")]:
            try:
                nltk.data.find(path_part)
            except LookupError:
                nltk.download(pkg, quiet=True)

        self.stop_words = set(stopwords.words(language))
        self.language = language

    def _tokenize_text(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            tokens = word_tokenize(text.lower(), language=self.language)
        except Exception:
            tokens = word_tokenize(text.lower())

        return [t for t in tokens if t.isalpha() and t not in self.stop_words]

    def process_documents(
        self,
        input_data: Union[Dict[str, str], List[Dict[str, str]]],
        top_n_words: int = 50
    ) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        is_single_doc_input = False
        if isinstance(input_data, dict):
            documents_input_list = [input_data]
            is_single_doc_input = True
        elif isinstance(input_data, list):
            documents_input_list = input_data
        else:
            raise ValueError("Input data must be a dict (for single doc) or a list of dicts (for multiple docs).")

        if not documents_input_list:
            return [] if is_single_doc_input else {}

        processed_docs_cache = []
        corpus_for_idf_vectorizer = []

        for i, doc_dict in enumerate(documents_input_list):
            filename = doc_dict.get("filename", f"unnamed_doc_{i}")
            content = doc_dict.get("content", "")

            tokens = self._tokenize_text(content)
            tf_counts = Counter(tokens)
            total_tokens_in_doc = sum(tf_counts.values())

            processed_docs_cache.append({
                "filename": filename,
                "tokens": tokens,
                "tf_counts": tf_counts,
                "total_tokens": total_tokens_in_doc
            })
            corpus_for_idf_vectorizer.append(" ".join(tokens))

        global_word_to_idf: Dict[str, float] = {}
        if any(doc_str.strip() for doc_str in corpus_for_idf_vectorizer):
            vectorizer = TfidfVectorizer(
                lowercase=False,
                stop_words=None,
                token_pattern=r"(?u)\b\w+\b",
                smooth_idf=True,
                use_idf=True
            )
            vectorizer.fit(corpus_for_idf_vectorizer)
            idf_values = vectorizer.idf_
            feature_names = vectorizer.get_feature_names_out()
            global_word_to_idf = dict(zip(feature_names, idf_values))

        results_by_filename: Dict[str, List[Dict[str, Any]]] = {}
        for doc_data in processed_docs_cache:
            filename = doc_data["filename"]
            tf_counts = doc_data["tf_counts"]
            total_tokens = doc_data["total_tokens"]
            
            current_doc_table_data: List[Dict[str, Any]] = []
            if not tf_counts:
                results_by_filename[filename] = []
                continue

            for word, tf_raw_count in tf_counts.items():
                idf_score = global_word_to_idf.get(word, 1.0)
                
                tf_normalized = (tf_raw_count / total_tokens) if total_tokens > 0 else 0.0
                
                current_doc_table_data.append({
                    "word": word,
                    "tf": tf_raw_count,                  # Сырая частота
                    "tf_norm": round(tf_normalized, 4), # Нормализованная частота
                    "idf": round(idf_score, 4)           # IDF
                })

            current_doc_table_data.sort(key=lambda x: (x["idf"], x["tf"]), reverse=True)

            results_by_filename[filename] = current_doc_table_data[:top_n_words]

        if is_single_doc_input:
            first_filename = documents_input_list[0].get("filename", "unnamed_doc_0")
            return results_by_filename.get(first_filename, [])
        else:
            return results_by_filename