""""
In orderr to process the 'Hellaswag' data for passing to the model, add the following code to this address:
./lm-evaluation-harness/lm_eval/evaluator.py ----> <<evaluate>> function ----> before 
                                                   resps = getattr(lm, reqtype)(cloned_reqs)

    1) Query: Main Context
    2) choices: options
    3) label: true answer
    4) tokenizer(Query) + tokenizer(choice)[:-2]: The input of the model
    5) Sum up the the likelihoods frm last 
columns = ['ind', 'Query', 'Choices', 'label']
ds = []
for idx, data in enumerate(requests['loglikelihood']):
            data_list = []
            data_list.append(idx)
            data_list.append(data.args[0])
            data_list.append(data.args[1])
            ds.append(data_list)

        df = pd.DataFrame(ds, columns=columns)
        df.to_json('/home/mohammad-m/TTT/RL/lm_eval_data/lambada_validation.json', orient="records", lines=True)
"""


import re
import csv
import torch
import string
import numpy as np
import pandas as pd
from collections import Counter
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline


dataset = pd.read_json("./lm_eval_data/lambada_validation.json", orient='records', lines=True)

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

openai_lambada_accuracy = []
openai_lambada_perplexity = []
standard_lambada_accuracy = []
standard_lambada_perplexity = []

for idx in tqdm(range(len(dataset)), desc="Processing"):
    loglokelihoods = []
    true_output = dataset['label'][idx]
    
    prompt = dataset['Query'][idx] + true_output
    tokenizer_choice = tokenizer(true_output, return_tensors='pt', add_special_tokens=False)
    tokenized = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    
    context_length = tokenized['input_ids'].shape[1] - 1
    choice_length = tokenizer_choice['input_ids'].shape[1]

    output = model(tokenized['input_ids']).logits
    multi_logits = F.log_softmax(output, dim=-1)

    logits = multi_logits[:, context_length - choice_length: context_length, :]
    greedy_tokens = logits.argmax(dim=-1)
    logits = torch.gather(logits, 2, tokenizer_choice['input_ids'].unsqueeze(-1)).squeeze(-1)  # [1, seq]

    ## <<Standard>> Lambada
    if dataset['ind'][idx] < 5153:
        if (greedy_tokens == tokenizer_choice['input_ids']).all(): standard_lambada_accuracy.append(1.0)
        else: standard_lambada_accuracy.append(0.0)
        standard_lambada_perplexity.append(float(logits.sum()))
    ## <<openai>> Lambada
    else:
        if (greedy_tokens == tokenizer_choice['input_ids']).all(): openai_lambada_accuracy.append(1.0)
        else: openai_lambada_accuracy.append(0.0)
        openai_lambada_perplexity.append(float(logits.sum()))
    



Total_standard_lambada_accuracy = np.sum(standard_lambada_accuracy) / len(standard_lambada_accuracy)
Total_standard_lambada_perplexity = np.exp(-np.mean(standard_lambada_perplexity))
Total_openai_lambada_accuracy = np.sum(openai_lambada_accuracy) / len(openai_lambada_accuracy)
Total_openai_lambada_perplexity = np.exp(-np.mean(openai_lambada_perplexity))
print("=========================================================================")
print(f"Standard Lambada Accuracy = {Total_standard_lambada_accuracy}")
print(f"Perplexity = {Total_standard_lambada_perplexity}")      
print("=========================================================================")
print(f"OpenAI Lambada Accuracy = {Total_openai_lambada_accuracy}")
print(f"OpenAI Lambada Perplexity = {Total_openai_lambada_perplexity}")      
print("=========================================================================")
print(f"Total Accuracy = {(Total_standard_lambada_accuracy + Total_openai_lambada_accuracy)/2}")
print(f"Total Lambada Perplexity = {(Total_standard_lambada_perplexity + Total_openai_lambada_perplexity)/2}") 
print("=========================================================================")
