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
                "You are an expert in API specifications. Based on the following retrieved documents from the API specification:\n\n"
                "{context}\n\n"
                "Please provide a clear and concise explanation of what the term '{input}' specifically refers to in the context of the provided API specifications. "
                "Ensure your answer is directly related to the provided documents and gives an accurate definition or explanation. "
                "Avoid unnecessary details, including repeating the query, and respond in Korean."
            )

    def get_template(self):
        if not self.prompt_template:
            raise ValueError("Prompt template has not been initialized.")
        return self.prompt_template
