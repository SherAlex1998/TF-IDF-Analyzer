# TF-IDF Analyzer

## Описание
TF-IDF Analyzer - это веб-приложение для анализа текстовых документов с использованием метода TF-IDF (Term Frequency-Inverse Document Frequency). Приложение позволяет загружать несколько текстовых файлов и анализировать частоту слов в них.

### Основные возможности
- Загрузка нескольких текстовых файлов
- Вычисление TF (Term Frequency) для каждого документа
- Вычисление IDF (Inverse Document Frequency) для всей коллекции документов
- Визуализация результатов в виде таблиц и графиков
- Нормализованные значения TF для корректного сравнения документов разной длины

### Технологии
- Backend: FastAPI
- Frontend: Streamlit
- Обработка текста: NLTK, scikit-learn
- Контейнеризация: Docker

## Запуск приложения

### Локальный запуск
1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
# или
venv\Scripts\activate  # для Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите бэкенд:
```bash
cd backend
uvicorn app.main:app --reload
```

4. В другом терминале запустите фронтенд:
```bash
cd frontend
streamlit run app.py
```

### Запуск через Docker
1. Соберите и запустите контейнеры:
```bash
docker-compose up --build
```

2. Откройте приложение в браузере:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000

---

# TF-IDF Analyzer

## Description
TF-IDF Analyzer is a web application for analyzing text documents using the TF-IDF (Term Frequency-Inverse Document Frequency) method. The application allows uploading multiple text files and analyzing word frequencies in them.

### Key Features
- Upload multiple text files
- Calculate TF (Term Frequency) for each document
- Calculate IDF (Inverse Document Frequency) for the entire document collection
- Visualize results in tables and charts
- Normalized TF values for correct comparison of documents with different lengths

### Technologies
- Backend: FastAPI
- Frontend: Streamlit
- Text Processing: NLTK, scikit-learn
- Containerization: Docker

## Running the Application

### Local Run
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # for Linux/Mac
# or
venv\Scripts\activate  # for Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend:
```bash
cd backend
uvicorn app.main:app --reload
```

4. In another terminal, start the frontend:
```bash
cd frontend
streamlit run app.py
```

### Running with Docker
1. Build and run containers:
```bash
docker-compose up --build
```

2. Open the application in your browser:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000

## Структура проекта

```
.
├── app/                   # Backend (FastAPI)
│   ├── api/               # API endpoints
│   └── services/          # Business logic
├── frontend/              # Frontend (Streamlit)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Технологии

- FastAPI
- Streamlit
- NLTK
- scikit-learn
- Docker 