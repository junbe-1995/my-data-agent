import pytest

from my_data_backend.config import config
from my_data_backend.modules.history_manager import HistoryManager
from langchain_openai import ChatOpenAI


@pytest.fixture
def history_manager():
    return HistoryManager()


@pytest.mark.asyncio
async def test_get_or_create_memory(history_manager):
    device_id = "test_device"
    llm = ChatOpenAI(
        model=config.OPENAI_LLM_MODEL_NAME, temperature=0.3, max_tokens=1000
    )

    memory = await history_manager.get_or_create_memory(device_id, llm)

    assert device_id in history_manager.user_history
    assert len(memory.chat_memory.messages) <= config.MAX_HISTORY_NUM

    # Test memory is the same for the same device_id
    memory_again = await history_manager.get_or_create_memory(device_id, llm)
    assert memory is memory_again
