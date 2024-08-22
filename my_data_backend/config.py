from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from os.path import dirname, join
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"

dotenv_path = join(dirname(__file__), "./", ".env")
load_dotenv(dotenv_path)


class Config(BaseSettings):
    STAGE: str = "dev"
    APP_NAME: str = "my_data_backend"
    DEBUG: bool = False
    LOG_LEVEL: str = "info"

    # Vectorstore
    VECTOR_SEARCH_TOP_K: int = 3

    # LLM
    OPENAI_LLM_MODEL_NAME: str = "gpt-4o-mini"
    MAX_HISTORY_NUM: int = 6

    # Pinecone
    USE_PINECONE: bool = False
    PINECONE_API_KEY: str = "your_pinecone_api_key"
    PINECONE_ENV_REGION: str = "your_pinecone_env_region"
    PINECONE_INDEX_NAME: str = "your_pinecone_index_name"

    # Cohere
    COHERE_MODEL_NAME: str = "embed-multilingual-v3.0"
    COHERE_API_KEY: str

    # Open AI
    OPENAI_API_KEY: str
    OPENAI_MAX_TOKENS: int = 1000

    # CLIP
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch16"


config = Config()
