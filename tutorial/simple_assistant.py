from dataclasses import dataclass

from flexrag.assistant import ASSISTANTS, AssistantBase
from flexrag.models import OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig


@dataclass
class SimpleAssistantConfig(DenseRetrieverConfig, OpenAIGeneratorConfig): ...


@ASSISTANTS("simple", config_class=SimpleAssistantConfig)
class SimpleAssistant(AssistantBase):
    def __init__(self, config: SimpleAssistantConfig):
        self.retriever = DenseRetriever(config)
        self.generator = OpenAIGenerator(config)
        return

    def answer(self, question: str) -> str:
        prompt = ChatPrompt()
        context = self.retriever.search(question)[0]
        prompt_str = ""
        for ctx in context:
            prompt_str += f"Question: {question}\nContext: {ctx.data['text']}"
        prompt.update(ChatTurn(role="user", content=prompt_str))
        response = self.generator.chat([prompt])[0][0]
        prompt.update(ChatTurn(role="assistant", content=response))
        return response
