from __future__ import annotations
from typing import Optional, Dict
import threading
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from my_data_backend.config import config


class HistoryManager:
    _instance: Optional[HistoryManager] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(HistoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = False
            self.user_history: Dict[str, ConversationSummaryMemory] = {}

    def initialize(self):
        with self._lock:
            if not self._initialized:
                self._initialized = True

    def get_or_create_memory(self, device_id: str, llm: ChatOpenAI):
        with self._lock:
            if device_id not in self.user_history:
                self.user_history[device_id] = ConversationSummaryMemory(
                    llm=llm, memory_key="chat_history"
                )

            memory = self.user_history[device_id]

            # 히스토리 개수를 제한
            while len(memory.chat_memory.messages) > config.MAX_HISTORY_NUM:
                memory.chat_memory.messages.pop(0)

        return memory
