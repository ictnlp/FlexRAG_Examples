from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from flexrag.assistant import ASSISTANTS, AssistantBase, SearchHistory
from flexrag.common_dataclass import RetrievedContext
from flexrag.models import GENERATORS, GenerationConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import RETRIEVERS, RetrieverConfig

GeneratorConfig = GENERATORS.make_config()


@dataclass
class ITRGAssistantConfig(GeneratorConfig, GenerationConfig, RetrieverConfig):
    max_iteration: int = 5
    retrieved_num: int = 5


@ASSISTANTS("itrg", "iter-retgen", config_class=ITRGAssistantConfig)
class ITRGAssistant(AssistantBase):
    prompt = ChatPrompt(
        system="Please write a detailed answer to the question based on the knowledge provided.",
    )

    def __init__(self, cfg: ITRGAssistantConfig) -> None:
        # load retriever
        self.retriever = RETRIEVERS.load(cfg)
        self.retriever.top_k = cfg.retrieved_num

        # load generator
        self.generator = GENERATORS.load(cfg)

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
        # prepare user prompt
        ctxs = [ctx.data.get("text", "") for ctx in contexts]
        ctx_str = " ".join(ctxs)
        user_prompt = f"Knowledge: {ctx_str}\nQuestion: {question}"

        # prepare prompt
        prompt = deepcopy(self.prompt)
        prompt.update(ChatTurn(role="user", content=user_prompt))

        # generate response
        response = self.generator.chat([prompt], generation_config=self.cfg)[0][0]
        return response, prompt
