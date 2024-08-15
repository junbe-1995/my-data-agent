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

    # LLM
    LLM_MODEL_NAME: str = "gpt-4o-mini"

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_ENV_REGION: str
    PINECONE_INDEX_NAME: str

    # Cohere
    COHERE_API_KEY: str

    # Open AI
    OPENAI_API_KEY: str


config = Config()
