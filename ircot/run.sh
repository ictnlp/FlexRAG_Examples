#!/bin/bash

MODEL_NAME=Qwen2-7B-Instruct
DATASET_NAME=nq
SPLIT=test
EXAMPLE_PATH="./ircot"


python -m flexrag.entrypoints.run_assistant \
    name=$DATASET_NAME \
    split=$SPLIT \
    user_module=$EXAMPLE_PATH \
    assistant_type=ircot \
    ircot_config.generator_type=openai \
    ircot_config.openai_config.model_name=$MODEL_NAME \
    ircot_config.openai_config.base_url=http://127.0.0.1:8000/v1 \
    ircot_config.do_sample=False \
    ircot_config.retriever_type='FlexRAG/wiki2021_atlas_contriever' \
    ircot_config.retrieved_num=5 \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
