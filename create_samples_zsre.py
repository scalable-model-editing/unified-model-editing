#here we create 10 samples of size 500 to sample the counterfact dataset from to visualize model editing effects

from dsets.zsre import MENDQADataset
from transformers import AutoTokenizer
import random
import json

random.seed(37)

if __name__ == '__main__':

    num_samples = 10 #number of samples
    write_flag = True
    sample_size = 10000
    output_filename = 'zsre_sampled_unique_' +  str(num_samples) + '_'  + str(sample_size) + '.json'


    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = MENDQADataset('data', tokenizer)


    #select indices with unique subjects in zsre
    all_subjects = {}
    for i in range(len(dataset)):
        subject = dataset.__getitem__(i)['requested_rewrite']['subject'].lower()
        
        if subject not in all_subjects:
            all_subjects[subject] = i
    

    selected_indices = list(all_subjects.values())

    #create samples for selected indices
    sampled_indices = {}
    for n in range(num_samples):
        random.shuffle(selected_indices)
        sampled_indices[n] = selected_indices[:sample_size]

    if write_flag:
        json_object = json.dumps(sampled_indices, indent=4)
        with open(output_filename , "w") as outfile:
            outfile.write(json_object)