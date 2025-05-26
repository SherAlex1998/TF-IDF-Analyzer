import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(
    page_title="TF-IDF Analyzer",
    page_icon="📊",
    layout="wide"
)

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

st.title("TF-IDF Analyzer")
st.write("Загрузите текстовые файлы для анализа TF-IDF")

uploaded_files = st.file_uploader("Выберите файлы", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    uploaded_files = list(reversed(uploaded_files))
    
    files = [('files', (file.name, file.getvalue())) for file in uploaded_files]
    response = requests.post(f'{BACKEND_URL}/api/process-files', files=files)
    
    if response.status_code == 200:
        data = response.json()
        
        first_doc = next(iter(data.values()))
        idf_df = pd.DataFrame(first_doc)[['word', 'idf']].drop_duplicates()
        idf_df.index = pd.RangeIndex(start=1, stop=len(idf_df) + 1)
        
        st.subheader(
            "IDF",
            help="Inverse Document Frequency (IDF) - обратная частота документа. "
                 "Показывает, насколько редким является слово во всей коллекции документов. "
                 "Чем выше значение IDF, тем более уникальным является слово."
        )
        st.dataframe(
            idf_df.style.format({
                'idf': '{:.4f}'
            }).set_properties(**{
                'text-align': 'left'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]},
                {'selector': 'td', 'props': [('text-align', 'left')]}
            ]),
            use_container_width=True
        )
        
        st.subheader(
            "Визуализация IDF",
            help="График показывает распределение IDF по словам. "
                 "Слова с высоким IDF встречаются редко в коллекции, "
                 "с низким - часто."
        )
        st.bar_chart(idf_df.set_index('word')['idf'])
        
        st.subheader(
            "TF",
            help="Term Frequency (TF) - частота слова в документе. "
                 "Показывает, как часто слово встречается в конкретном документе. "
                 "Чем выше значение TF, тем чаще слово используется в документе."
        )
        
        filenames = [file.name for file in uploaded_files]
        tabs = st.tabs(filenames)
        
        for tab, filename in zip(tabs, filenames):
            with tab:
                doc_data = data[filename]
                tf_df = pd.DataFrame(doc_data)[['word', 'tf', 'tf_norm']]
                tf_df.index = pd.RangeIndex(start=1, stop=len(tf_df) + 1)
                
                st.write(f"Таблица TF для {filename}")
                st.dataframe(
                    tf_df.style.format({
                        'tf': '{:.0f}',
                        'tf_norm': '{:.4f}'
                    }).set_properties(**{
                        'text-align': 'left'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'left')]},
                        {'selector': 'td', 'props': [('text-align', 'left')]}
                    ]),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Сырой TF для {filename}")
                    st.bar_chart(tf_df.set_index('word')['tf'])
                
                with col2:
                    st.write(f"Нормированный TF для {filename}")
                    st.bar_chart(tf_df.set_index('word')['tf_norm'])
    else:
        st.error("Произошла ошибка при обработке файлов") 