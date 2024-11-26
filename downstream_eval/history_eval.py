import os
import sys
import json
import matplotlib.pyplot as plt

sys.path.append('/home/akshatgupta/KnowledgeEditing/memit')
from experiments.summarize import summarize

def get_stats(stats_dict, sorted_history):
    for num_edits, hist_folder in sorted_history:
        hist_path = data_location + hist_folder
        stats = summarize(abs_path = hist_path)

        for stat in stats_dict:
            stats_dict[stat][num_edits] = stats[stat][0]

    return stats_dict

def plot_stats(stats_dict, filename):
    for stat in stats_dict:
        sorted_dict = sorted(stats_dict[stat].items(), key=lambda item: item[0])
        
        x, y = [], []
        for edit_num, correct in sorted_dict:
            x.append(edit_num)
            y.append(correct)

        plt.plot(x,y, label = stat)
        plt.legend()

    plt.savefig(save_location + filename)
    plt.close()


if __name__ == '__main__':
    algo = 'MEMIT'
    run = 'run_010'
    save_location = 'downstream_eval/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/'

    hist_folders = {}
    for folder in os.listdir(data_location):
        if 'history' in folder:
            num_edits = int(folder.split('_')[-1])
            hist_folders[num_edits] = folder

    sorted_history = sorted(hist_folders.items(), key=lambda item: item[0])

    #success stats
    acc_stats = {
        'post_rewrite_success' : {},
        'post_paraphrase_success' : {},
        'post_neighborhood_success' : {},
        'post_rewrite_acc' : {},
        'post_paraphrase_acc' : {},
        'post_neighborhood_acc' : {},        
    }

    acc_stats = get_stats(acc_stats, sorted_history)
    plot_stats(acc_stats, algo + '_history_acc.png')


    #difference stats
    acc_stats = {
        'post_rewrite_diff' : {},
        'post_paraphrase_diff' : {},
        'post_neighborhood_diff' : {},     
    }

    acc_stats = get_stats(acc_stats, sorted_history)
    plot_stats(acc_stats, algo + '_history_diff.png')


    #entropy stat
    acc_stats = {
        'post_ngram_entropy' : {},
        'post_reference_score' : {},   
    }

    acc_stats = get_stats(acc_stats, sorted_history)
    plot_stats(acc_stats, algo + '_history_gen.png')


    #overall score
    acc_stats = {
        'post_score' : {}, 
    }

    acc_stats = get_stats(acc_stats, sorted_history)
    plot_stats(acc_stats, algo + '_history_score.png')



