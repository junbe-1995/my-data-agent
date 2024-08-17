from __future__ import annotations
from typing import Optional, Dict
import asyncio
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from my_data_backend.config import config


class HistoryManager:
    _instance: Optional[HistoryManager] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HistoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = False
            self.user_history: Dict[str, ConversationSummaryMemory] = {}
            self.history_locks: Dict[str, asyncio.Lock] = {}

    def initialize(self):
        if not self._initialized:
            self._initialized = True

    async def get_or_create_memory(self, device_id: str, llm: ChatOpenAI):
        # device_id 별로 고유한 락을 생성하여 동시 접근 제어
        if device_id not in self.history_locks:
            self.history_locks[device_id] = asyncio.Lock()

        async with self.history_locks[device_id]:
            if device_id not in self.user_history:
                self.user_history[device_id] = ConversationSummaryMemory(
                    llm=llm, memory_key="chat_history"
                )

            memory = self.user_history[device_id]

            # 히스토리 개수를 제한
            while len(memory.chat_memory.messages) > config.MAX_HISTORY_NUM:
                memory.chat_memory.messages.pop(0)

        return memory
