from dataclasses import dataclass
from typing import Optional

from flexrag.assistant import (
    ASSISTANTS,
    AssistantBase,
    SearchHistory,
)
from flexrag.models import GENERATORS, GenerationConfig
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig, RetrievedContext

GeneratorConfig = GENERATORS.make_config()


@dataclass
class IRCoTAssistantConfig(GeneratorConfig, GenerationConfig, DenseRetrieverConfig):
    max_iteration: int = 5


@ASSISTANTS("ircot", config_class=IRCoTAssistantConfig)
class IRCoTAssistant(AssistantBase):
    prompt = (
        "You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. "
        "This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. "
        "Your task is to generate one thought for current step, DON'T generate the whole thoughts at once! "
        'If you reach what you believe to be the final step, start with "So the answer is:".\n\n'
        "Wikipedia Title: Kurram Garhi\n"
        "Kurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. "
        "Its population is approximately 35000. Barren hills are near this village. "
        "This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\n"
        "Wikipedia Title: 2001–02 UEFA Champions League second group stage\n"
        "Eight winners and eight runners- up from the first group stage were drawn into four groups of four teams, "
        "each containing two group winners and two runners- up. "
        "Teams from the same country or from the same first round group could not be drawn together. "
        "The top two teams in each group advanced to the quarter- finals.\n\n"
        "Wikipedia Title: Satellite tournament\n"
        "A satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\n"
        "Wikipedia Title: Trojkrsti\n"
        "Trojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\n"
        "Wikipedia Title: Telephone numbers in Ascension Island\n"
        "Country Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\n"
        "Question: Are both Kurram Garhi and Trojkrsti located in the same country?\n"
        "Thought: Kurram Garhi is located in the country of Pakistan. "
        "Trojkrsti is located in the country of Republic of Macedonia. "
        "Thus, they are not in the same country. So the answer is: no.\n\n"
    )

    def __init__(self, cfg: IRCoTAssistantConfig) -> None:
        # load retriever
        self.retriever = DenseRetriever(cfg)

        # load generator
        self.generator = GENERATORS.load(cfg)

        # set configs
        self.max_iter = cfg.max_iteration
        self.cfg = cfg
        self.cfg.stop_str = ["\n", "."]
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

            # generate CoT
            prompt_str = f"{self.prompt}\n\n"
            for ctx in ctxs:
                prompt_str += f"Wikipedia Title: {ctx.data.get('title')}\n{ctx.data.get('text')}\n\n"
            prompt_str += f"Question: {question}\n"
            prompt_str += "Thought: "
            thought = self.generator.generate([prompt_str], self.cfg)[0][0]

            # update query
            if "So the answer is:" in thought:
                response = thought.split("So the answer is:")[1].strip()
                break
            response = query = thought
        return response, ctxs, {"history": history}
