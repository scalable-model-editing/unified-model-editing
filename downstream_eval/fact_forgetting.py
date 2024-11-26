import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/akshatgupta/KnowledgeEditing/model-editing')
from experiments.summarize import summarize
from useful_functions import save_data


if __name__ == '__main__':
    #plotting hyperparameters
    x_tick_size = 22
    y_tick_size = 22
    x_lim = 1000
    y_lim = 110
    axis_fontsize = 24
    legend_fontsize = 16

    algo = 'ROME'
    run = 'run_006'
    sample_num = '3'
    save_location = 'downstream_eval/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/'
    #data_location = 'results/' + algo + '/' + 'run_015' + '/'

    #get order of edits
    indices_filename = 'counterfact_sampled_unique_10_20391.json'
    #indices_filename = 'zsre_sampled_unique_10_10720.json'
    f = open(indices_filename)
    sampled_indices = json.load(f)


    #Store index where correct fact was stored
    all_cases = {}
    correct_facts = 0
    for e, element_index in enumerate(sampled_indices[sample_num]):
        #if e < 1100:
        #    continue

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


    forget_len = []
    forget_indices = []
    for case_num in all_cases:
        if 'forget_index' in all_cases[case_num]:
            forget_len.append(all_cases[case_num]['forget_index'] - all_cases[case_num]['edit_num'])
            forget_indices.append(all_cases[case_num]['forget_index']) 


    #plotting forgetting length histogram
    forget_len = np.array(forget_len)
    sns.histplot(forget_len,  bins=10)
    plt.title('MEAN :' + str(np.mean(forget_len)) + '  STD : ' + str(np.std(forget_len)) )
    plt.savefig(save_location + algo + '_forget_len_dist.png')
    plt.close()

    #plotting forget indices
    forgetting_indices = forget_indices
    sns.histplot(forgetting_indices,  bins=10)
    plt.savefig(save_location + algo + '_forget_locations.png')
    plt.close()


    ###CALCULATING CUMULATIVE FORGETTING
    correct_so_far = []
    edited_so_far = []
    forgotten = 0
    for e, case_num in enumerate(sampled_indices[sample_num]):
        if e >= max_total_edits:
            continue

        edited_so_far.append(e+1)
        correct_so_far.append(all_cases[case_num]['correct_so_far'])


    forget_indices_sorted = sorted(forget_indices)

    forgotten_so_far = []
    cum_sum = 0
    for e in range(max_total_edits):
        facts_forgotten_at_this_index = forget_indices_sorted.count(e+1)
        cum_sum += facts_forgotten_at_this_index
        
        forgotten_so_far.append(cum_sum)



    #plotting rememberence percentage
    x, y, y_correct = [], [], []
    for edits, correct_edits, forgotten in zip(edited_so_far, correct_so_far, forgotten_so_far):
        x.append(edits)
        y.append( (forgotten/edits) * 100)

        if correct_edits == 0:
            y_correct.append(0)
        else:
            y_correct.append( (forgotten / correct_edits) * 100  )


    plt.plot(x, y, label = 'forgot %')
    plt.plot(x, y_correct, label = 'forgot over correct %')
    plt.legend()
    plt.savefig(save_location + algo + '_forget_cummulative.png')
    plt.close()



    run_title = {}
    plt.figure(figsize=(6.5, 4.5))
    #plotting rememberence percentage over correct edits for paper

    plt.plot(x, y_correct, label = 'Facts Forgotten', linewidth = 3, color = 'r')
    plt.legend(fontsize=legend_fontsize)
    #plt.legend(loc='upper left', bbox_to_anchor=(0.4, 1.28), ncol=1, fontsize=14)
    plt.xlabel('Number of Edits', fontsize=axis_fontsize)
    plt.ylabel('% Facts Forgotten', fontsize=axis_fontsize)
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.tick_params(axis='x', labelsize=x_tick_size)
    plt.tick_params(axis='y', labelsize=y_tick_size)
    plt.tight_layout()
    if run in run_title:
        plt.savefig(save_location + algo + '_forget_cummulative_correct_' + run_title[run] + '.png')
    else:
        plt.savefig(save_location + algo + '_forget_cummulative_correct.png')
    plt.close()

    save_data(algo + sample_num + '.pkl', y_correct)