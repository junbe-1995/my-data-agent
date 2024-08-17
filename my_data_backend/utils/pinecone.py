from __future__ import annotations
from typing import Optional
import threading
import pinecone
from my_data_backend.config import config


class PineconeSingleton:
    _instance: Optional[PineconeSingleton] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PineconeSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = False

    def initialize(self):
        with self._lock:
            if not self._initialized:
                pinecone.init(
                    api_key=config.PINECONE_API_KEY,
                    environment=config.PINECONE_ENV_REGION,
                )
                self._initialized = True
