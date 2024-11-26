import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

sys.path.append('/home/akshatgupta/KnowledgeEditing/model-editing')
from useful_functions import save_data

if __name__ == '__main__':
    x_tick_size = 22
    y_tick_size = 22
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 24
    legend_fontsize = 16


    algo = 'b-ROME'
    run = 'run_024'
    save_location = 'downstream_eval/plots/' + algo + '/' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/'


    distance_files = {}
    entropy_files = {}


    normalized_entropy = []
    entropy = []
    distances = {}
    for filename in os.listdir(data_location):
        file_loc = data_location + filename
        if 'case' in filename:
            with open(file_loc, "r") as f:
                data = json.load(f)


            #collect variables
            normalized_entropy.append(data['normalized_entropy'])
            entropy.append(data['entropy'])
            entropy_files[filename] = data['normalized_entropy']

            for layer in data['distance']:
                if layer not in distances:
                    distances[layer] = []
                    distance_files[layer] = {}

                distances[layer].append(data['distance'][layer])
                distance_files[layer][filename] = data['distance'][layer]

    
    for layer in distances:
        plt.scatter(distances[layer], normalized_entropy, color = 'r')

        plt.xlabel('Normalized Distance', fontsize=axis_fontsize)
        plt.ylabel('Normalized Entropy', fontsize=axis_fontsize)
        plt.tick_params(axis='x', labelsize=x_tick_size)
        plt.tick_params(axis='y', labelsize=y_tick_size)
        plt.tight_layout()

        # Format the x-axis ticks
        ax = plt.gca()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        ax.xaxis.set_major_formatter(formatter)

        plt.savefig(save_location + 'disabling_normalized_entropy_layer_{}.png'.format(layer))
        plt.close()


    for layer in distances:
        plt.scatter(distances[layer], entropy, color = 'r')

        plt.xlabel('Normalized Distance', fontsize=axis_fontsize)
        plt.ylabel('Entropy', fontsize=axis_fontsize)
        plt.tick_params(axis='x', labelsize=x_tick_size)
        plt.tick_params(axis='y', labelsize=y_tick_size)
        plt.tight_layout()

        # Format the x-axis ticks
        ax = plt.gca()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        ax.xaxis.set_major_formatter(formatter)

        plt.savefig(save_location + 'disabling_entropy_layer_{}.png'.format(layer))
        plt.close()
                

    sorted_entropy = sorted(entropy_files.items(), key=lambda item: item[1])

    print('PRINTING ENTROPY')
    for filename, entropy in sorted_entropy[:10]:
        print(filename, entropy)
    print()

    for layer in distance_files:
        distance_f = sorted(distance_files[layer].items(), key=lambda item: item[1], reverse=True)
        
        print('PRINTING ENTROPY Layer', layer)
        for filename, distance in distance_f[:10]:
            print(filename, distance)
        print()

    print('TOTAL DONE:', len(sorted_entropy))