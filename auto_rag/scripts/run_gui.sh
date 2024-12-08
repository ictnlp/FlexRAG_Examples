#!/bin/bash

MODEL_NAME=AutoRAG
MODEL_URL=127.0.0.1
RETRIEVER="<your dense retriever path>"

CUDA_VISIBLE_DEVICES=0,1,2,3 python webui.py \
    --main_model {model_name} \
    --main_model_url {main_model_url} \
    --dense_corpus_path {dense_corpus_path} \
    --dense_index_path {dense_index_path}
