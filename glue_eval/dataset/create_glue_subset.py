from datasets import load_dataset
from useful_functions import save_data
import random

random.seed(37)


classwise_size = 100
# for dataset_name in ['sst2', 'mrpc', 'cola', 'rte']:
#     dataset = load_dataset("glue", dataset_name)
#     eval_dataset = dataset['validation']

#     classwise = {}
#     finalized_subset = []

#     for example in eval_dataset:
#         if example['label'] not in classwise:
#             classwise[example['label']] = [example]
#         else:
#             classwise[example['label']].append(example)

#     for label in classwise:
#         random.shuffle(classwise[label])
#         finalized_subset += classwise[label][:classwise_size]

#     random.shuffle(finalized_subset)
#     save_data(dataset_name + '.pkl', finalized_subset)


dataset = load_dataset("cais/mmlu", "all")
eval_dataset = dataset['validation']

classwise = {}
finalized_subset = []

for example in eval_dataset:
    if example['answer'] not in classwise:
        classwise[example['answer']] = [example]
    else:
        classwise[example['answer']].append(example)

for label in classwise:
    random.shuffle(classwise[label])
    finalized_subset += classwise[label][:classwise_size]

random.shuffle(finalized_subset)
save_data( 'mmlu.pkl', finalized_subset)
    
    
