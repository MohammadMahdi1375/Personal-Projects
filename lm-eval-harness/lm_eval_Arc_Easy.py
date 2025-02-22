import csv
import torch
import string
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline

##### load_dataset('allenai/ai2_arc', 'ARC-Challenge')
dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy', split="test")
print(dataset)

"""device = "cuda:0"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device,
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)


csv_file = "Arc_Easy_Reward.csv"
header = ['question', 'answerKey', 'A', 'B', 'C', 'D']
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
cnt = 0
cnt_total = 0
for i, data in enumerate(tqdm(dataset)):
    #prompt = data['question'] + ' ' + data['choices']['text'] + ' ' + data['choices']['labels']
    
    prompt = data['question'] + ' ' + str(data['choices'])

    if (i == 0):
        print(prompt)

    response1 = 'A'
    response2 = 'B'
    response3 = 'C'
    response4 = 'D'

    #response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

    conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
    conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]
    conv3 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response3}]
    conv4 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response4}]

    # Format and tokenize the conversations
    # If you use `tokenize=False` with `apply_chat_template` and `tokenizer()` to tokenize the conversation,
    # remeber to remove the duplicated BOS token.
    conv1_tokenized = rm_tokenizer.apply_chat_template(conv1, tokenize=True, return_tensors="pt").to(device)
    conv2_tokenized = rm_tokenizer.apply_chat_template(conv2, tokenize=True, return_tensors="pt").to(device)
    conv3_tokenized = rm_tokenizer.apply_chat_template(conv3, tokenize=True, return_tensors="pt").to(device)
    conv4_tokenized = rm_tokenizer.apply_chat_template(conv4, tokenize=True, return_tensors="pt").to(device)

    # Get the reward scores
    scores = {}
    with torch.no_grad():
        scores['A'] = rm(conv1_tokenized).logits[0][0].item()
        scores['B'] = rm(conv2_tokenized).logits[0][0].item()
        scores['C'] = rm(conv3_tokenized).logits[0][0].item()
        scores['D'] = rm(conv4_tokenized).logits[0][0].item()

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([data['question'], data['answerKey'], scores['A'], scores['B'], scores['C'], scores['D']])

        answer = data['answerKey']
        if (data['answerKey'] == '1'): answer = 'A'
        elif (data['answerKey'] == '2'): answer = 'B'
        elif (data['answerKey'] == '3'): answer = 'C'
        elif (data['answerKey'] == '4'): answer = 'D'

        if (answer == 'A' or answer == 'B' or answer == 'C' or answer == 'D'):
            cnt_total += 1
            if (max(list(scores.values())) != scores[answer]):
                print(data['question'], scores)
                cnt += 1
        
print(cnt, cnt_total)
print("Done")
"""


"""pipeline_model = pipeline(
                "text-generation",
                model="meta-llama/Llama-3.2-1B",
                tokenizer="meta-llama/Llama-3.2-1B",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
                model_kwargs=None
            )"""
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
#dataset = dataset.select(range(5))

model_results = []
model_results_norm = []
label_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, '1':0, '2':1, '3':2, '4':3}
for data in tqdm(dataset):
    prompt1 = "Question: " + data['question'] + "\nAnswer:"
    tokenized_input1 = tokenizer(prompt1, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda')

    ##### Storing the loglikelihood of each member in the list 
    loglokelihoods = []
    choice_lens = []
    true_output = label_map[data['answerKey']]
    for choice in data['choices']['text']:
        ##### Answer Tokenization
        choice_lens.append(len(choice))
        prompt2 = ' ' + choice
        prompt2 = prompt2.rstrip(string.punctuation)
        tokenized_input2 = tokenizer(prompt2, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda')
        
        ##### lengths
        GT_len = tokenized_input2.shape[1]
        model_inp_len = tokenized_input1.shape[1] + GT_len - 1

        ##### Model Prompt generation
        tokenized_input = torch.cat((tokenized_input1, tokenized_input2[:, :GT_len-1]), dim=1)

        ##### Model Output
        output = model(tokenized_input).logits
        multi_logits = F.log_softmax(output, dim=-1)

        logits = multi_logits[:, model_inp_len - GT_len: model_inp_len, :]
        logits = torch.gather(logits, 2, tokenized_input2.unsqueeze(-1)).squeeze(-1)  # [1, seq]

        loglokelihoods.append(float(logits.sum()))

    pred_output_choice = np.argmax(loglokelihoods)
    pred_output_choice_norm = np.argmax(list(np.array(loglokelihoods) / np.array(choice_lens)))

    if pred_output_choice == true_output: model_results.append(1.0)
    else: model_results.append(0.0)
    if pred_output_choice_norm == true_output: model_results_norm.append(1.0)
    else: model_results_norm.append(0.0)

    
print("=========================================================================")
print(f"Accuracy = {np.sum(model_results) / len(model_results)}")        
print(f"Accuracy_Norm = {np.sum(model_results_norm) / len(model_results_norm)}")        
print("=========================================================================")

