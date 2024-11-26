import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/akshatgupta/KnowledgeEditing_crtx/unified-model-editing')
from useful_functions import save_data

if __name__ == '__main__':
    x_tick_size = 22
    y_tick_size = 22
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 24
    legend_fontsize = 16

    full_distance_metrics = ["preservation_distance", "new_edit_distance", "old_edit_distance",
                        "delta_norm", "new_weights_norm", "original_weights_norm"]
    distance_metrics = ["delta_norm", "new_weights_norm", "original_weights_norm"]

    metric_names = ['correct', 'f1', 'mcc', 'invalid']
    task_names = ['sst', 'mrpc', 'cola', 'rte']
    
    glue_eval = {'distance':{}, 'objective_distances':{}}
    for task in task_names:
        glue_eval[task] = {}
        for metric in metric_names:
            glue_eval[task][metric] = {}


    algo = 'EMMET'
    run = 'run_010'
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
                edit_num = int(data['edit_num']) + 1

            if 'base' not in filename:
                #plot distance data
                for layer in data['distance_from_original']:
                    if layer not in glue_eval['distance']:
                        glue_eval['distance'][layer] = {}

                    glue_eval['distance'][layer][edit_num] = data['distance_from_original'][layer]


                #plot other distance data
                for layer in data['objective_distances']:
                    if layer not in glue_eval['objective_distances']:
                        glue_eval['objective_distances'][layer] = {}

                    for metric in data['objective_distances'][layer]:
                        if metric not in glue_eval['objective_distances'][layer]:
                            glue_eval['objective_distances'][layer][metric] = {}

                        glue_eval['objective_distances'][layer][metric][edit_num] = data['objective_distances'][layer][metric]


            #load task specific data
            for task in task_names:
                if task in data:
                    for metric in metric_names:
                        glue_eval[task][metric][int(edit_num)] = data[task][metric]



    #plot metrics individual with number of edits
    for metric in metric_names:
        #plt.figure(figsize=(6.5, 5.5))

        x_labels = []
        y_values = {}
        width = 1 / (len(task_names) + 1)

        for task in task_names:
            sorted_dict = sorted(glue_eval[task][metric].items(), key=lambda item: item[0])

            if not x_labels:
                x_labels = [a for a, b in sorted_dict]
            y_values[task] = [b for a, b in sorted_dict]


        positions = np.array(x_labels)
        fig, ax = plt.subplots()
        for t, task in enumerate(task_names):
            bar = ax.bar(positions + t*width, y_values[task], width, label=task)


        ax.set_xlabel('Number of Edits', fontsize=axis_fontsize)
        ax.set_ylabel(metric.upper(), fontsize=axis_fontsize)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)
        ax.legend()
        plt.tight_layout()


        plt.savefig(save_location + algo + '_' + 'glue_' + metric + '.png')
        plt.close()




    print(glue_eval['objective_distances'])


    width = 1 / (len(distance_metrics) + 1)
    #plot metrics individual with number of edits
    for layer in glue_eval['objective_distances']:
        #plt.figure(figsize=(6.5, 5.5))

        x_labels = []
        y_values = {}
        for metric in distance_metrics:
            sorted_dict = sorted(glue_eval['objective_distances'][layer][metric].items(), key=lambda item: item[0])


            if not x_labels:
                x_labels = [a for a, b in sorted_dict]
            y_values[metric] = [b if b != None else 0 for a, b in sorted_dict]
            y_values[metric] = [y if y < 1000 else 1000 for y in y_values[metric]]


        positions = np.array(x_labels)
        fig, ax = plt.subplots()
        for t, task in enumerate(distance_metrics):
            bar = ax.bar(positions + t*width, y_values[task], width, label=task)


        ax.set_xlabel('Number of Edits', fontsize=axis_fontsize)
        ax.set_ylabel(metric.upper(), fontsize=axis_fontsize)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)
        ax.legend()
        plt.tight_layout()


        plt.savefig(save_location + algo + '_' + 'layer_' + str(layer) + '_distance_metrics.png')
        plt.close()

