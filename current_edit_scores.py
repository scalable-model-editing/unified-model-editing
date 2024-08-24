###This file measures the efficacy of making the current edit given a larger number of edits has been made in the past


import os
import json
import math
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    average_array = []

    for i in range(len(data)):
        window = data[max(i - window_size + 1, 0) : i + 1 ]
        window_avg = sum(window) / len(window)
        average_array.append(window_avg)

    return average_array


if __name__ == '__main__':
    x_tick_size = 22
    y_tick_size = 22
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 24
    legend_fontsize = 16


    metrics = {
        'rewrite_prompts_probs' : [], 
        'paraphrase_prompts_probs' : [], 
        'neighborhood_prompts_probs' : []
    }

    algo = 'FT'
    run = 'run_003'
    sample_num = '0'
    save_location = 'downstream_eval/plots/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/'
    bucket_size = 5
    window_size = 10

    #get order of edits
    indices_filename = 'data/counterfact_sampled_unique_mcf_10_20000.json'
    f = open(indices_filename)
    sampled_indices = json.load(f)

    batch_size = 1
    for e, element_index in enumerate(sampled_indices[sample_num]):
        filename = '{}_{}_edits-case_{}.json'.format(batch_size, e, element_index)

        file_loc = data_location + filename

        if not os.path.exists(file_loc):
            break

        with open(file_loc, "r") as f:
            data = json.load(f)

        try:
            for metric in metrics:
                if metric in ['rewrite_prompts_probs', 'paraphrase_prompts_probs']:
                    success = [element['post_target_new_prob'] < element['post_target_true_prob'] for element in data['post'][metric]]
                else:
                    success = [element['post_target_new_prob'] > element['post_target_true_prob'] for element in data['post'][metric]]

                value = sum(success)/len(success)
                metrics[metric].append(value)
        except:
            pass


    #making individual bar plots
    for metric in metrics:
        x, y = [], []
        for i in range( math.ceil(len(metrics[metric])//bucket_size) ):
            x.append(i)

            start_index = i * bucket_size
            end_index = min((i + 1) * bucket_size, len(metrics[metric]))
            y.append(sum(metrics[metric][start_index: end_index]))

        plt.bar(x, y)
        plt.savefig(save_location + algo + '_score_' + metric + '.png')
        plt.close()


    metric_colors = {
        'rewrite_prompts_probs' : 'k', 
        'paraphrase_prompts_probs' : 'b', 
        'neighborhood_prompts_probs' : 'r'
    }
    metric_labels = {
        'rewrite_prompts_probs' : 'Efficacy Score', 
        'paraphrase_prompts_probs' : 'Paraphrase Score', 
        'neighborhood_prompts_probs' : 'Neighborhood Score'
    }
    run_title = {}
    #making overall plot
    plt.figure(figsize=(6.5, 6))
    for metric in metrics:
        x, y = [], []
        for i in range( math.ceil(len(metrics[metric])//bucket_size) ):
            x.append(i * bucket_size)

            start_index = i * bucket_size
            end_index = min((i + 1) * bucket_size, len(metrics[metric]))

            y.append((sum(metrics[metric][start_index: end_index]) / bucket_size) * 100)

        y_avg = moving_average(y, window_size)

        plt.plot(x, y, linestyle = '--', color = metric_colors[metric], linewidth = 0.2)
        plt.plot(x, y_avg, color = metric_colors[metric], label = metric_labels[metric], linewidth = 2)

    plt.legend(loc='upper left', bbox_to_anchor=(0.45, 1.28), ncol=1, fontsize=legend_fontsize)
    plt.xlabel('Number of Edits', fontsize=axis_fontsize)
    plt.ylabel('Edit Accuracy', fontsize=axis_fontsize)
    #plt.xlim(0, x_lim)
    #plt.title(run_title[run])
    #plt.suptitle(run_title[run], y=0.0, verticalalignment='bottom')
    plt.tick_params(axis='x', labelsize=x_tick_size)
    plt.tick_params(axis='y', labelsize=y_tick_size)
    plt.tight_layout()
    if run in run_title:
        plt.savefig(save_location + algo + '_editing_score_' + run_title[run] +  '.png')
    else:
        plt.savefig(save_location + algo + '_editing_score.png')
    plt.close()

    
    

        

        

