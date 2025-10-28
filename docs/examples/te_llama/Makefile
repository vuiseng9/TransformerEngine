
install-hf:
	pip install -U transformers accelerate peft datasets

00-run-hf-bf16:
	python 00_bf16_hf_llama2_7b.py

01-run-te-bf16:
	python 01_bf16_te_llama2_7b.py

02-run-te-fp8:
	python 02_fp8_te_llama2_7b.py

03-run-te-mxfp8:
	python 03_mxfp8_te_llama2_7b.py