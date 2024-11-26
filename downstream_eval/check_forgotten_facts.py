import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/akshatgupta/KnowledgeEditing/memit')
from experiments.summarize import summarize


if __name__ == '__main__':
    algo = 'ROME'
    run = 'run_022'
    sample_num = '4'
    save_location = 'downstream_eval/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/'

    #get order of edits
    indices_filename = 'counterfact_sampled_10_15000.json'
    f = open(indices_filename)
    sampled_indices = json.load(f)


    #Store index where correct fact was stored
    all_cases = {}
    correct_facts = 0
    for e, element_index in enumerate(sampled_indices[sample_num]):

        filename = 'case_{}.json'.format(str(element_index))
        file_loc = data_location + filename

        if not os.path.exists(file_loc):
            max_total_edits = e 
            break

        with open(file_loc, "r") as f:
            data = json.load(f)

        if data['post']['rewrite_prompts_correct'][0]:#model learnt the fact correctly
            correct_facts += 1
        
        case_num = data['case_id']
        all_cases[case_num] = {
                        'edit_success': data['post']['rewrite_prompts_correct'][0],
                        'edit_num': e+1, 
                        'correct_so_far': correct_facts
                        }


    #read history evaluation folders in order
    hist_folders = {}
    for folder in os.listdir(data_location):
        if 'history' in folder:
            num_edits = int(folder.split('_')[-1])
            hist_folders[num_edits] = folder

    sorted_history = sorted(hist_folders.items(), key=lambda item: item[0])

    for num_edits, hist_folder in sorted_history:
        hist_path = data_location + hist_folder + '/'

        #get sorted edits list
        edit_history = {}
        for filename in os.listdir(hist_path):
            if '.json' in filename:
                case_location = hist_path + filename
                edit_num = int(filename.split('_')[0])
                case_num = int(filename.replace('.json', '').split('_')[-1])

                with open(case_location, "r") as f:
                    data = json.load(f)

                if not data['post']['rewrite_prompts_correct'][0] and all_cases[case_num]['edit_success'] and 'forget_index' not in all_cases[case_num]:
                    all_cases[case_num]['forget_index'] = num_edits

                if 'foget_index' in all_cases[case_num] and data['post']['rewrite_prompts_correct'][0]:#previouly forgotten and now remembered again
                    print('YAYYYY')
