from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from flexrag.assistant import AssistantBase
from flexrag.data import IterableDataset
from flexrag.models import GenerationConfig, OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig

from prompts import Knowledge_Prompt


@dataclass
class Config:
    main_model_config: OpenAIGeneratorConfig = field(default_factory=OpenAIGeneratorConfig)  # fmt: skip
    retriever_config: DenseRetrieverConfig = field(default_factory=DenseRetrieverConfig)
    data_path: str = MISSING
    max_iter: int = 10
    elicit_max_iter: int = 5
    max_passages: int = 2
    verbose: bool = False


def dense_test(
    main_model: OpenAIGenerator,
    dense_retriever: DenseRetriever,
    dataset: IterableDataset,
    max_iter: int = 10,
    elicit_max_iter: int = 5,
    max_passages: int = 2,
    verbose: bool = False,
):
    trace = []
    for i in dataset:
        queries = [i["question"]]
        retrieved_ids = []
        prompt = ChatPrompt(
            system="Answer the question by retrieving external knowledge. Extract useful information from each retrieved document. If the information is insufficient or irrelevant, refine your query and search again until you are able to answer the question.",
            history=[
                ChatTurn(role="user", content="Question: " + i["question"].strip())
            ],
        )

        # start retrieval iteration
        current_iter = 0
        first_model_output = None
        while max_iter > 0:
            if verbose:
                print("input", prompt)

            # generate action
            first_model_output = main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(do_sample=False, max_new_tokens=200),
            )[0][0].strip()

            if "Query:".lower() in first_model_output.lower():
                queries = [first_model_output.split("Query:")[-1].strip()]
                current_iter += 1
            elif "final answer" in first_model_output.lower():
                prompt.update(ChatTurn(role="assistant", content=first_model_output))
                break
            else:
                print("Exception: Follow Failed")
                print(prompt)
                print(first_model_output)

            prompt.update(ChatTurn(role="assistant", content=first_model_output))

            document = None
            queries[0] = queries[0].replace("[Dense]", "").strip()
            documents = []
            retrieval_results = dense_retriever.search(queries[0])[0]

            for result in retrieval_results:
                if result.data["id"] not in retrieved_ids:
                    retrieved_ids.append(result.data["id"])
                    documents.append(result.data["text"].split("\n")[-1])
                if len(documents) >= max_passages:
                    break
            document = " ".join(documents)

            prompt.update(
                ChatTurn(
                    role="user",
                    content=f"Retrieved Document_{current_iter}: {document.strip()}",
                )
            )

            max_iter -= 1

        first_model_output = ""
        if max_iter == 0:
            first_model_output = main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(temperature=0.0, max_new_tokens=150),
            )[0][0].strip()

            prompt.update(ChatTurn(role="assistant", content=first_model_output))

        max_iter = elicit_max_iter
        while "Refined Query:" in first_model_output and max_iter > 0:
            current_iter += 1
            query = first_model_output.split("Refined Query:")[-1].strip()

            document_prompt = Knowledge_Prompt.format(i["question"], query)

            document = main_model.generate(
                prefixes=document_prompt,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_new_tokens=200,
                    stop_str=["<|eot_id|>", "\n"],
                ),
            )[0][0].strip()

            prompt.update(
                ChatTurn(
                    role="user",
                    content=f"Retrieved Document_{current_iter}: {document.strip()}",
                )
            )

            first_model_output = main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(
                    do_sample=False,
                    max_new_tokens=150,
                ),
            )[0][0].strip()

            prompt.update(
                ChatTurn(
                    role="assistant",
                    content=first_model_output,
                )
            )
            max_iter -= 1

        trace.append(
            {
                "question": i["question"],
                "prompt": prompt,
                "golden_answers": i["golden_answers"],
            }
        )
        if "answer_id" in i.keys():
            trace[-1]["answer_id"] = i["answer_id"]

    return "success"


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    # prepare AutoRAG
    main_model = OpenAIGenerator(cfg.main_model_config)
    retriever = DenseRetriever(cfg.retriever_config)
    testset = IterableDataset(cfg.data_path)

    # start retrieval
    dense_test(
        main_model=main_model,
        dense_retriever=retriever,
        dataset=testset,
        max_iter=cfg.max_iter,
        elicit_max_iter=cfg.elicit_max_iter,
        max_passages=cfg.max_passages,
        verbose=cfg.verbose,
    )
    return


if __name__ == "__main__":
    main()
