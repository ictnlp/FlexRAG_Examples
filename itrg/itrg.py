from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from flexrag.assistant import (
    ASSISTANTS,
    PREDEFINED_PROMPTS,
    AssistantBase,
    SearchHistory,
)
from flexrag.models import GENERATORS, GenerationConfig
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig, RetrievedContext
from flexrag.prompt import ChatPrompt, ChatTurn

GeneratorConfig = GENERATORS.make_config()


@dataclass
class ITRGAssistantConfig(GeneratorConfig, GenerationConfig, DenseRetrieverConfig):
    max_iteration: int = 2
    prompt_path: Optional[str] = None


@ASSISTANTS("itrg", "iter-retgen", config_class=ITRGAssistantConfig)
class ITRGAssistant(AssistantBase):
    def __init__(self, cfg: ITRGAssistantConfig) -> None:
        # load retriever
        self.retriever = DenseRetriever(cfg)

        # load generator
        self.generator = GENERATORS.load(cfg)

        # load prompt
        if cfg.prompt_path is None:
            self.prompt = PREDEFINED_PROMPTS["shortform_with_context"]
        else:
            self.prompt = ChatPrompt.from_json(cfg.prompt_path)

        # set configs
        self.max_iter = cfg.max_iteration
        self.cfg = cfg
        return

    def answer(
        self, question: str
    ) -> tuple[str, Optional[list[RetrievedContext]], Optional[dict]]:
        history: list[SearchHistory] = []
        query = question
        for _ in range(self.max_iter):
            # retrieve
            ctxs = self.retriever.search([query])[0]
            history.append(SearchHistory(question, ctxs))

            # generate
            response, _ = self.answer_with_contexts(question, ctxs)

            # update query
            query = f"{query} {response}"
        return response, ctxs, {"history": history}

    def answer_with_contexts(
        self, question: str, contexts: list[RetrievedContext] = []
    ) -> tuple[str, ChatPrompt]:
        # prepare system prompts
        prompt = deepcopy(self.prompt)

        # prepare user prompt
        usr_prompt = ""
        for n, context in enumerate(contexts):
            ctx = ""
            for field_name, field_value in context.data.items():
                ctx += f"{field_name}: {field_value}\n"
            usr_prompt += f"Context {n + 1}: {ctx}\n\n"
        usr_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=usr_prompt))

        # generate response
        response = self.generator.chat([prompt], generation_config=self.cfg)[0][0]
        return response, prompt
