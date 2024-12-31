from flexrag.models import HFEncoderConfig, OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig


def main():
    # Initialize the retriever
    retriever_cfg = DenseRetrieverConfig(database_path="path_to_database", top_k=1)
    retriever_cfg.query_encoder_config.encoder_type = "hf"
    retriever_cfg.query_encoder_config.hf_config = HFEncoderConfig(
        model_path="facebook/contriever"
    )
    retriever = DenseRetriever(retriever_cfg)

    # Initialize the generator
    generator = OpenAIGenerator(
        OpenAIGeneratorConfig(
            model_name="gpt-4o-mini", api_key="your_openai_key", do_sample=False
        )
    )

    # Run your RAG application
    prompt = ChatPrompt()
    while True:
        query = input("Please input your query (type `exit` to exit): ")
        if query == "exit":
            break
        context = retriever.search(query)[0]
        prompt_str = ""
        for ctx in context:
            prompt_str += f"Question: {query}\nContext: {ctx.data['text']}"
        prompt.update(ChatTurn(role="user", content=prompt_str))
        response = generator.chat(prompt)
        prompt.update(ChatTurn(role="assistant", content=response))
        print(response)
    return


if __name__ == "__main__":
    main()
