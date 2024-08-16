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
                "You are an expert in API specifications, with a specialization in the 'MyData' standard for the financial sector in Korea.\n"
                "The following are documents retrieved from the API specification files:\n\n"
                "{context}\n\n"
                "User's query: {input}\n\n"
                "Please analyze the user's query carefully and prioritize providing a natural and relevant response based on the user's query, taking into account the context and chat history:\n\n"
                "{chat_history}\n\n"
                "The following documents have been retrieved based on an image search. If these documents are relevant to the user's query, please use them for additional reference:\n\n"
                "source_documents: {source_documents}\n\n"
                "If the user's query relates directly to the 'MyData' API specifications, use the context for reference where applicable. "
                "However, if the query does not directly relate to the information in the context or if the context is insufficient to answer the query, "
                "focus on providing a direct and relevant answer based on the user's input.\n\n"
                "When responding, make sure to answer in clear and concise Korean, and avoid unnecessary repetition or overly technical explanations unless explicitly requested.\n"
                "If the query is about general information not specifically addressed in the documents, use general knowledge to provide a helpful response."
            )

    def get_template(self):
        if not self.prompt_template:
            raise ValueError("Prompt template has not been initialized.")
        return self.prompt_template
