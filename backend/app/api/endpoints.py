from fastapi import APIRouter, UploadFile, File
from app.services.tfidf_service import TFIDFService
from typing import List

router = APIRouter()
tfidf_service = TFIDFService()

@router.post("/process-files")
async def process_files(files: List[UploadFile] = File(...)):
    documents = []
    for file in files:
        content = await file.read()
        documents.append({
            'filename': file.filename,
            'content': content.decode('utf-8')
        })
    
    result = tfidf_service.process_documents(documents)
    
    return result 