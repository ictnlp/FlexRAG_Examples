from argparse import ArgumentParser, Namespace

from flexrag.models import HFEncoderConfig, OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig


def main(args: Namespace):
    # Initialize the retriever
    retriever_cfg = DenseRetrieverConfig(
        database_path=args.database_path, top_k=1, index_type="faiss"
    )
    retriever_cfg.query_encoder_config.encoder_type = "hf"
    retriever_cfg.query_encoder_config.hf_config = HFEncoderConfig(
        model_path=args.encoder_name
    )
    retriever = DenseRetriever(retriever_cfg)

    # Initialize the generator
    generator = OpenAIGenerator(
        OpenAIGeneratorConfig(
            model_name=args.openai_model_name,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
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
        response = generator.chat([prompt])[0][0]
        prompt.update(ChatTurn(role="assistant", content=response))
        print(response)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="facebook/contriever",
        help="The name or path to the encoder",
    )
    parser.add_argument(
        "--openai_model_name",
        type=str,
        default="gpt-4o-mini",
        help="The name of the OpenAI generator",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        required=True,
        help="The API Key of your OpenAI account.",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=None,
        help="The base url to the OpenAI service. Default to None.",
    )
    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
        help="The path to the retriever database",
    )
    args = parser.parse_args()
    main(args)
