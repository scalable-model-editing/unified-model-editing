import os
import sys
import json
import matplotlib.pyplot as plt
import torch

sys.path.append('/home/akshatgupta/KnowledgeEditing_local/unified-model-editing')
from useful_functions import save_data

if __name__ == '__main__':
    x_tick_size = 22
    y_tick_size = 22
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 24
    legend_fontsize = 16


    algo = 'EMMET'
    run = 'run_048'
    save_location = 'downstream_eval/plots/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/glue_eval/'

    spectral_analysis = {'svd_upd':{}, 'svd_final':{}, 'new_weights_norm': {}, 'z_norms': {'delta':{}, 'init_norm':{}, 'final_norm':{}}}
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
                for layer in data['objective_distances']:
                    if layer == 'z_norms':
                        for key in data['objective_distances']['z_norms']:
                            for subkey in data['objective_distances']['z_norms'][key]:
                                if edit_num in spectral_analysis['z_norms'][subkey]:
                                    spectral_analysis['z_norms'][subkey][edit_num].append(data['objective_distances']['z_norms'][key][subkey])
                                else:
                                    spectral_analysis['z_norms'][subkey][edit_num] = [data['objective_distances']['z_norms'][key][subkey]]
                        continue
                    
                    elif layer not in spectral_analysis['svd_upd']:
                        spectral_analysis['svd_upd'][layer] = {}
                        spectral_analysis['svd_final'][layer] = {}
                        spectral_analysis['new_weights_norm'][layer] = {}

                    spectral_analysis['svd_upd'][layer][edit_num] = data['objective_distances'][layer]['svd_upd']
                    spectral_analysis['svd_final'][layer][edit_num] = data['objective_distances'][layer]['svd_final']
                    spectral_analysis['new_weights_norm'][layer][edit_num] = data['objective_distances'][layer]['new_weights_norm']

    break_flag = False
    break_index = 440

    top_n = 1600##number of top singular values to plots
    for matrix_type in ['svd_final']:
        for layer in spectral_analysis[matrix_type]:
            sorted_dict = sorted(spectral_analysis[matrix_type][layer].items(), key=lambda item: item[0])

            for i in range(-1, top_n):
                x, y = [], []
                for edit_num, svd in sorted_dict:
                    x.append(edit_num)
                    y.append(svd[i])
                    if break_flag and edit_num == break_index:
                        break

                plt.plot(x, y, label=str(i))

            plt.legend()
            plt.savefig(save_location + matrix_type + '_' + layer + '.png')
            plt.close()

    ##plot energy
    for matrix_type in ['svd_final']:
        for layer in spectral_analysis[matrix_type]:
            sorted_dict = sorted(spectral_analysis[matrix_type][layer].items(), key=lambda item: item[0])

            x, y = [], []
            for edit_num, svd in sorted_dict:
                x.append(edit_num)
                svd = torch.tensor(svd)
                ratio = svd[0] / torch.norm(svd)
                y.append(ratio)
                if break_flag and edit_num == break_index:
                    break

            plt.plot(x, y, label='energy ratio-' + str(layer))
        plt.legend()
        plt.savefig(save_location + matrix_type + '_energy.png')
        plt.close()

    ##plot condition number
    for matrix_type in ['svd_final']:
        for layer in spectral_analysis[matrix_type]:
            sorted_dict = sorted(spectral_analysis[matrix_type][layer].items(), key=lambda item: item[0])

            x, y = [], []
            for edit_num, svd in sorted_dict:
                x.append(edit_num)
                ratio = svd[0] / svd[-1]
                y.append(ratio)
                if break_flag and edit_num == break_index:
                    break

            plt.plot(x, y, label='condition number-' + str(layer))
        plt.legend()
        plt.savefig(save_location + matrix_type + '_condition.png')
        plt.close()

    ##entropy
    for matrix_type in ['svd_final']:
        for layer in spectral_analysis[matrix_type]:
            sorted_dict = sorted(spectral_analysis[matrix_type][layer].items(), key=lambda item: item[0])

            x, y = [], []
            for edit_num, svd in sorted_dict:
                x.append(edit_num)
                svd = torch.tensor(svd)
                p = torch.nn.functional.softmax(svd)
                entropy = -torch.sum(p * torch.log(p))
                y.append(entropy)
                if break_flag and edit_num == break_index:
                    break

            plt.plot(x, y, label='entropy-' + str(layer))
        plt.legend()
        plt.savefig(save_location + matrix_type + '_entropy.png')
        plt.close()


    for matrix_type in ['new_weights_norm']:
        for layer in spectral_analysis[matrix_type]:
            sorted_dict = sorted(spectral_analysis[matrix_type][layer].items(), key=lambda item: item[0])

            x, y = [], []
            for edit_num, svd in sorted_dict:
                x.append(edit_num)
                y.append(round(svd, 2))
                if break_flag and edit_num == break_index:
                        break

            plt.plot(x, y, label='norm-' + str(layer))
        plt.legend()
        plt.savefig(save_location + matrix_type + '.png')
        plt.close()

    for matrix_type in ['z_norms']:
        for layer in spectral_analysis[matrix_type]:
            sorted_dict = sorted(spectral_analysis[matrix_type][layer].items(), key=lambda item: item[0])

            x, y = [], []
            for edit_num, svd in sorted_dict:
                x.append(edit_num)
                y.append(sum(svd)/len(svd))
                if break_flag and edit_num == break_index:
                        break

            plt.plot(x, y, label=layer)
        plt.legend()
        plt.savefig(save_location + matrix_type + '.png')
        plt.close()