#!/bin/bash

MODEL_NAME=Qwen2-7B-Instruct


python -m vllm.entrypoints.openai.api_server \                                                                                                                                                                                        (base)
    --model $MODEL_NAME \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code
