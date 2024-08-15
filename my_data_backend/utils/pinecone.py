import pinecone
from my_data_backend.config import config


class PineconeSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if not self._instance._initialized:
            pinecone.init(
                api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV_REGION
            )
            self._instance._initialized = True
