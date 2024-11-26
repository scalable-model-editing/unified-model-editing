import os
import sys
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.append('/home/akshatgupta/KnowledgeEditing/model-editing')
from experiments.summarize import summarize
from dsets.counterfact import CounterFactDataset

def dataloader(tokenizer, prompt, sentence, loss_ignore_index):
    prompt_ids = tokenizer.encode(prompt)
    sentence_ids = tokenizer.encode(sentence)
    labels = [loss_ignore_index] * len(prompt_ids) +  sentence_ids[len(prompt_ids):]
    
    input_ids = sentence_ids[:-1]
    label_ids = labels[1:]

    return torch.tensor(input_ids), torch.tensor(label_ids)

def evaluate(model, tokenizer, criterion, prompt, sentence, loss_ignore_index):
    model.eval()  # turn on train mode
    total_loss = 0.
    start_time = time.time()

    input_ids, label_ids = dataloader(tokenizer, prompt, sentence, loss_ignore_index)

    input_ids = input_ids.reshape(1, -1)
    label_ids = label_ids.reshape(1, -1)

    with torch.no_grad():
        input_ids, labels= input_ids.to(model.device), label_ids.to(model.device)

        output = model(input_ids=input_ids)
        output = output['logits']

        output_flat = output.contiguous().view(-1, model.config.vocab_size)
        target_flat = labels.contiguous().view(-1)

        loss = criterion(output_flat, target_flat)
        total_loss += loss.item()


    ms_per_batch = (time.time() - start_time)
    cur_loss = total_loss
    ppl = math.exp(cur_loss)

    return ppl

def moving_average(data, window_size):
    average_array = []

    for i in range(len(data)):
        window = data[max(i - window_size + 1, 0) : i + 1 ]
        window_avg = sum(window) / len(window)
        average_array.append(window_avg)

    return average_array


if __name__ == '__main__':
    dataset = CounterFactDataset('data')
    loss_ignore_index = -1
    model_name = 'gpt2'
    #model_name = '/data/akshat/lexi-models/Llama-2-7b-hf'
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    criterion = nn.CrossEntropyLoss(ignore_index=loss_ignore_index)

    algo = 'MEND'
    run = 'run_007'
    sample_num = '4'
    save_location = 'downstream_eval/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/'
    bucket_size = 5
    window_size = 10

    #get order of edits
    indices_filename = 'counterfact_sampled_unique_10_20391.json'
    #indices_filename = 'counterfact_sampled_10_15000.json'
    f = open(indices_filename)
    sampled_indices = json.load(f)


    #Store index where correct fact was stored
    all_cases = {}
    correct_facts = 0
    
    gen_entropy = []
    first_entropy = True
    for e, element_index in enumerate(sampled_indices[sample_num]):
        datapoint = dataset.__getitem__(element_index)

        filename = 'case_{}.json'.format(str(element_index))
        file_loc = data_location + filename

        if not os.path.exists(file_loc):
            max_total_edits = e 
            break

        with open(file_loc, "r") as f:
            data = json.load(f)

        collected_perplexity = []
        for prompt, sentence in zip(datapoint['generation_prompts'], data['post']['text']):
            perplexity = evaluate(model, tokenizer, criterion, prompt, sentence, loss_ignore_index)
            if perplexity < 5e4:
                collected_perplexity.append(perplexity)

        gen_entropy.append(sum(collected_perplexity) / len(collected_perplexity))
        print(e)


    #making overall plot
    plt.figure(figsize=(6.5, 6))
    x, y = [], []
    for i in range( math.ceil(len(gen_entropy)//bucket_size) ):
        x.append(i * bucket_size)

        start_index = i * bucket_size
        end_index = min((i + 1) * bucket_size, len(gen_entropy))

        y.append(sum(gen_entropy[start_index: end_index]) / bucket_size)

    y_avg = moving_average(y, window_size)
    plt.plot(x, y, linestyle = '--', color = 'r', linewidth = 1)
    plt.plot(x, y_avg, label = 'Generation Perplexity', color = 'r', linewidth = 4)
    #plt.ylim(0,2000)
    plt.legend()
    plt.savefig(save_location + algo + '_generation_perplexity.png')
    plt.close()