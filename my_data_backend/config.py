from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from os.path import dirname, join


dotenv_path = join(dirname(__file__), "./", ".env")
load_dotenv(dotenv_path)


class Config(BaseSettings):
    STAGE: str = "dev"
    APP_NAME: str = "my_data_backend"
    DEBUG: bool = False
    REGION: str = "kr"

    # Vectorstore
    VECTOR_SEARCH_TOP_K: int = 3

    # LLM
    OPENAI_LLM_MODEL_NAME: str = "gpt-4o-mini"
    MAX_HISTORY_NUM: int = 6

    # Pinecone
    USE_PINECONE: bool = False
    PINECONE_API_KEY: str
    PINECONE_ENV_REGION: str
    PINECONE_INDEX_NAME: str

    # Cohere
    COHERE_MODEL_NAME: str = "multilingual-22-12"
    COHERE_API_KEY: str

    # Open AI
    OPENAI_API_KEY: str

    # CLIP
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch16"
    KMP_DUPLICATE_LIB_OK: str
    TOKENIZERS_PARALLELISM: str


config = Config()
