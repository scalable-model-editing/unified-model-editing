import os
import sys
import json
import matplotlib.pyplot as plt
import util.nethook as nethook
from transformers import AutoModelForCausalLM
from attrdict import AttrDict
import torch

sys.path.append('/home/akshatgupta/KnowledgeEditing_local/unified-model-editing')
from useful_functions import save_data

if __name__ == '__main__':
    model_name = 'gpt2-xl'#'/data/akshat/models/Llama-2-7b-hf'
    model = AutoModelForCausalLM.from_pretrained(model_name)

    algo = 'FT'
    run = 'run_003'

    hparams_filename = 'hparams/FT/gpt2-xl_unconstr.json'
    #hparams_filename = 'hparams/' + algo + '/gpt2-xl.json'
    f = open(hparams_filename)
    hparams = AttrDict(json.load(f))
    layers_edited = list(hparams.layers)

    original_norms = {}
    for layer_num in layers_edited:
        edited_layer = nethook.get_module(model, hparams.rewrite_module_tmp.format(layer_num))
        original_norms[str(layer_num)] = torch.norm(edited_layer.weight).item()

    x_tick_size = 22
    y_tick_size = 22
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 24
    legend_fontsize = 16

    metric_names = ['correct', 'f1', 'mcc', 'invalid']
    task_names = ['sst', 'mmmlu', 'nli', 'rte']
    
    glue_eval = {'distance':{}}
    for task in task_names:
        glue_eval[task] = {}
        for metric in metric_names:
            glue_eval[task][metric] = {}

    save_location = 'downstream_eval/plots/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/glue_eval/'

    for filename in os.listdir(data_location):
        file_loc = data_location + filename
        if 'glue' in filename:
            with open(file_loc, "r") as f:
                data = json.load(f)

            if 'base' in filename:
                edit_num = 0
            else:
                edit_num = data['edit_num'] + 1

                #plot distance data
                for layer in data['distance_from_original']:
                    if layer not in glue_eval['distance']:
                        glue_eval['distance'][layer] = {}
                        glue_eval['distance'][layer][0] = original_norms[layer]

                    glue_eval['distance'][layer][edit_num] = data['objective_distances'][layer]['new_weights_norm']

            for task in task_names:
                if task in data:
                    for metric in metric_names:
                        glue_eval[task][metric][int(edit_num)] = data[task][metric]


    task_dict = {'sst':'SST2', 'mmmlu':'MMLU', 'rte':'COLA', 'nli':'NLI'}
    run_title = {}
    task_colors = {'sst':'r', 'mmmlu':'b', 'rte':'g', 'nli':'k'}
    #plot metrics individual with number of edits
    for metric in metric_names:
        plt.figure(figsize=(6.5, 5.5))
        for task in task_names:
            sorted_dict = sorted(glue_eval[task][metric].items(), key=lambda item: item[0])

            x, y = [], []
            for edit_num, correct in sorted_dict:
                x.append(edit_num)
                if metric in ['f1']:
                    y.append(correct * 100)
                else:
                    y.append(correct)

            plt.plot(x,y, label = task_dict[task], linewidth =3, color=task_colors[task])

        plt.legend(fontsize=legend_fontsize)
        plt.xlabel('Number of Edits', fontsize=axis_fontsize)
        if metric == 'correct':
            metric = 'accuracy'
        plt.ylabel(metric.upper(), fontsize=axis_fontsize)
        #plt.xlim(0, x_lim)
        plt.ylim(0, y_lim)
        plt.tick_params(axis='x', labelsize=x_tick_size)
        plt.tick_params(axis='y', labelsize=y_tick_size)
        plt.tight_layout()

        if run in run_title:
            plt.savefig(save_location + algo + '_' + 'glue_' + metric + '_' + run_title[run] + '.png')
        else:
            plt.savefig(save_location + algo + '_' + 'glue_' + metric + '.png')
        plt.close()

    #plot distance as a function of number of edits
    metric = 'distance'
    x_store = []
    y_store = []
    for l, layer in enumerate(glue_eval[metric]):
        sorted_dict = sorted(glue_eval[metric][layer].items(), key=lambda item: item[0])

        x, y = [], []
        for edit_num, correct in sorted_dict:
            x.append(edit_num)
            y.append(correct)

        x_store.append(x)
        y_store.append(y)

        if 'transformer' in layer:
            layer = layer.split('.')[2]

        if l == 0:
            plt.plot(x,y, linewidth =3, color = 'r', label = 'Layer ' + str(int(layer) + 1))
        else:
            plt.plot(x,y, linewidth =3, label = 'Layer ' + str(int(layer) + 1))
        
    plt.legend(fontsize=legend_fontsize)
    plt.xlabel('Number of Edits', fontsize=axis_fontsize)
    plt.ylabel('Normalized Distance', fontsize=axis_fontsize)
    #plt.ylim(0, 1000)
    plt.tick_params(axis='x', labelsize=x_tick_size)
    plt.tick_params(axis='y', labelsize=y_tick_size)
    plt.tight_layout()

    if run in run_title:
        plt.savefig(save_location + algo + '_' + 'distance_' + run_title[run] + '.png')
    else:
        plt.savefig(save_location + algo + '_' + 'distance.png')
    plt.close()
            
    #print(len(y))
    #print(y)
    #ave_data(algo + sample_num + '_distance.pkl', [x_store,y_store])
    #plot glue performance as a function of number of edits
    #plot metrics individual with number of edits


    for layer_num in original_norms:

        for metric in metric_names:
            plt.figure(figsize=(6.5, 5.5))
            for task in task_names:
                sorted_dict = sorted(glue_eval[task][metric].items(), key=lambda item: item[0])

                x, y = [], []
                for index, (edit_num, correct) in enumerate(sorted_dict):
                    x.append(glue_eval['distance'][str(layer_num)][edit_num])
                    if metric in ['f1', 'accuracy']:
                        y.append(correct * 100)
                    else:
                        y.append(correct)

                plt.plot(x,y, label = task_dict[task], linewidth =3, color=task_colors[task])

            plt.legend(fontsize=legend_fontsize)
            plt.xlabel('Matrix Norm', fontsize=axis_fontsize)
            if metric == 'correct':
                metric = 'accuracy'
            plt.ylabel(metric.upper(), fontsize=axis_fontsize)
            #plt.xlim(100, 200)
            plt.ylim(0, y_lim)
            plt.tick_params(axis='x', labelsize=x_tick_size)
            plt.tick_params(axis='y', labelsize=y_tick_size)
            plt.tight_layout()

            plt.savefig(save_location + algo + '_' + 'glue_' + metric + '_distance_' + layer_num + '.png')
            plt.close()