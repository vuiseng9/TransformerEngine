On 1xb200, llama2-7b

hf bf16
Average time taken per step: 177 milliseconds

te bf16
Average time taken per step: 135 milliseconds

te fp8
Average time taken per step: 159 milliseconds

te mxfp8
Average time taken per step: 174 milliseconds


how to integrate new recipe:
modify te utils hyperparameters
te/docs/examples/te_llama/utils.py
modify accelerate utils
/opt/venv/lib/python3.12/site-packages/accelerate/utils/transformer_engine.py