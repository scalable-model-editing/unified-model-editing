import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/akshatgupta/KnowledgeEditing/model-editing')
from experiments.summarize import summarize


def get_distribution(input_list):
    frequency = {}

    for element in input_list:
        if element not in frequency:
            frequency[element] = 1
        else:
            frequency[element] += 1

    #convert frequency to distribution
    for element in frequency:
        frequency[element] /= len(input_list)

    return frequency

def get_entropy(input_dict):
    entropy = 0
    probabilities = input_dict.values()
    
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def moving_average(data, window_size):
    average_array = []

    for i in range(len(data)):
        window = data[max(i - window_size + 1, 0) : i + 1 ]
        window_avg = sum(window) / len(window)
        average_array.append(window_avg)

    return average_array


if __name__ == '__main__':
    algo = 'ROME'
    run = 'run_016'
    sample_num = '1'
    save_location = 'downstream_eval/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/'
    bucket_size = 5
    window_size = 10

    #get order of edits
    #indices_filename = 'counterfact_sampled_unique_10_20391.json'
    indices_filename = 'zsre_sampled_unique_10_10720.json'
    #indices_filename = 'counterfact_sampled_10_15000.json'
    f = open(indices_filename)
    sampled_indices = json.load(f)


    #Store index where correct fact was stored
    all_cases = {}
    correct_facts = 0
    
    gen_entropy = []
    first_entropy = True
    for e, element_index in enumerate(sampled_indices[sample_num]):
        filename = 'case_{}.json'.format(str(element_index))
        file_loc = data_location + filename

        if not os.path.exists(file_loc):
            max_total_edits = e 
            break

        with open(file_loc, "r") as f:
            data = json.load(f)

        words = []
        for sentence in data['post']['text']:
            words.extend(sentence.lower().replace('.', '').split(' '))

        distribution = get_distribution(words)
        entropy = get_entropy(distribution)

        '''if first_entropy:
            first_norm = math.log2(len(distribution))
            first_entropy = False'''

        normalized_entropy = entropy / math.log2(len(distribution))
        #normalized_entropy = entropy / first_norm

        gen_entropy.append(normalized_entropy)


    #making overall plot
    plt.figure(figsize=(6.5, 6))
    x, y = [], []
    for i in range( math.ceil(len(gen_entropy)//bucket_size) ):
        x.append(i * bucket_size)

        start_index = i * bucket_size
        end_index = min((i + 1) * bucket_size, len(gen_entropy))

        y.append(sum(gen_entropy[start_index: end_index]) / bucket_size)

    y_avg = moving_average(y, window_size)
    plt.plot(x, y, linestyle = '--', color = 'r', linewidth = 2)
    plt.plot(x, y_avg, label = 'Generation Entropy', color = 'k', linewidth = 4)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig(save_location + algo + '_generation_entropy.png')
    plt.close()