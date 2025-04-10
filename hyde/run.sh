#!/bin/bash

MODEL_NAME=Qwen2-7B-Instruct
BASE_URL=http://127.0.0.1:8000/v1
DATASET_NAME=nq
SPLIT=test


python -m flexrag.entrypoints.run_assistant \
    name=$DATASET_NAME \
    split=$SPLIT \
    assistant_type=modular \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name=$MODEL_NAME \
    modular_config.openai_config.base_url=$BASE_URL \
    modular_config.do_sample=False \
    modular_config.retriever_type=hyde \
    modular_config.hyde_config.generator_type=openai \
    modular_config.hyde_config.openai_config.model_name=$MODEL_NAME \
    modular_config.hyde_config.openai_config.base_url=$BASE_URL \
    modular_config.hyde_config.database_path=wiki2021_atlas_contriever \
    modular_config.hyde_config.index_type=faiss \
    modular_config.hyde_config.query_encoder_config.encoder_type=hf \
    modular_config.hyde_config.query_encoder_config.hf_config.model_path=facebook/contriever-msmarco \
    modular_config.hyde_config.query_encoder_config.hf_config.device_id=[0] \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
