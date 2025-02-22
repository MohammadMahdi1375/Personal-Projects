# LM Evaluation Harness
lm-evaluation-harness is an open-source benchmarking tool developed by EleutherAI to evaluate large language models (LLMs) on a variety of tasks. It provides a standardized way to assess the performance of LLMs across multiple datasets and metrics. It supports Multiple Benchmark Datasets:
- Language modeling (e.g., WikiText, Pile)
- Question answering (e.g., ARC, MMLU)
- Common sense reasoning (e.g., HellaSwag)
- Mathematical reasoning (e.g., GSM8K)

The interesting point regarding lm-evaluaiton-harness is that it works with popular LLMs, including Hugging Face models, OpenAI’s API, and EleutherAI’s GPT models but you will run into so many challenges when you want to apply it to your custom model. All this motivated me to start writing my own evaluation script based on the steps that are done in lm-evaluation-harness. It is pretty user-friendly and you have to import your custom model and the script will automatically evaluate your model.


