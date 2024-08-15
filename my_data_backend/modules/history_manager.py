from typing import Optional, Dict
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from my_data_backend.config import config


class HistoryManager:
    _instance: Optional["HistoryManager"] = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HistoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.user_history: Dict[str, ConversationSummaryMemory] = {}
        self._initialized = True

    def get_or_create_memory(self, device_id: str, llm: ChatOpenAI):
        if device_id not in self.user_history:
            self.user_history[device_id] = ConversationSummaryMemory(
                llm=llm, memory_key="chat_history"
            )

        memory = self.user_history[device_id]

        # 히스토리 개수를 제한
        while len(memory.chat_memory.messages) > config.MAX_HISTORY_NUM:
            memory.chat_memory.messages.pop(0)

        return memory
