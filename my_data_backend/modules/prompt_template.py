from langchain_core.prompts import ChatPromptTemplate
from typing import Optional


class PromptTemplateSingleton:
    _instance: Optional["PromptTemplateSingleton"] = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PromptTemplateSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.prompt_template = None
        self._initialized = True

    def initialize(self):
        if not self.prompt_template:
            self.prompt_template = ChatPromptTemplate.from_template(
                "You are an expert in API specifications, specifically focusing on the 'MyData' standard for the financial sector in Korea.\n"
                "Based on the following documents retrieved from the API specification files:\n\n"
                "{context}\n\n"
                "Please provide a clear and concise explanation in response to the user's query: '{input}'. "
                "Be sure to consider the conversation history to understand the context:\n\n"
                "{chat_history}\n\n"
                "Make sure your answer addresses the user's request within the context of previous messages, offering more detailed information if needed. "
                "Avoid unnecessary user's query repetition, and respond in Korean."
            )

    def get_template(self):
        if not self.prompt_template:
            raise ValueError("Prompt template has not been initialized.")
        return self.prompt_template
