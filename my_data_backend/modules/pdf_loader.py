import os
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

cache_dir = "./cache"
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(cache_dir, "documents_cache.pkl")


def load_pdfs_from_folder_with_cache_and_chunking(folder_path: str):
    if os.path.exists(cache_file):
        print("캐시에서 문서 로드 중...")
        with open(cache_file, "rb") as f:
            documents = pickle.load(f)
        print("문서 로드 완료!")
    else:
        documents = []
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )  # 청킹 크기 조정

        for i, filename in enumerate(pdf_files, start=1):
            print(f"{i}/{len(pdf_files)} PDF 파일 로드 중: {filename}")
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            raw_documents = loader.load()
            chunked_documents = text_splitter.split_documents(raw_documents)
            documents.extend(chunked_documents)

        print("PDF 파일 로드 완료, 캐시에 저장 중...")
        with open(cache_file, "wb") as f:
            pickle.dump(documents, f)
        print("캐시에 문서 저장 완료!")

    return documents


async def load_pdfs_async(folder_path: str):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        print("비동기 PDF 로드 시작...")
        documents = await loop.run_in_executor(
            pool, load_pdfs_from_folder_with_cache_and_chunking, folder_path
        )
        print("비동기 PDF 로드 완료!")
    return documents
