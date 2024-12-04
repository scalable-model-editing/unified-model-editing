# EMMET is now able to perform batched edits for batch sizes of 10k matching the performance of MEMIT

## News 

Our latest study on Model Editing using Llama-3 just released - [Is Bigger Edit Batch Size Always Better? - An Empirical Study on Model Editing with Llama-3](https://arxiv.org/abs/2405.00664)

# A Unified Framework for Model Editing

This repository unifies model-editing algorithms under the same conceptual framework of **preservation-memorization** and also under the same code-base. We use a common set of functions to calculate the key and value vectors and explicitly apply the update equations of ROME and MEMIT in the same file. 

We also introduce the EMMET through this repository and add its update equations in this unified codebase.

This code also allows for distributing edits across multiple layers using the MEMIT edit-distribution algorithm for ROME, EMMET and MEMIT.

We hope that such a unified code-base simplifies both conceptual and implementational understanding of these model editing algorithms.

## Installation
We work off of the [MEMIT](https://github.com/kmeng01/memit) codebase, so we'll reference the same installation procedures here: 
"We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`."


## Running the experiments
To evaluate EMMET, run the following command:

```python
python experiments/evaluate_unified_editing.py \
--alg_name=EMMET \
--num_edits=4 \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--ds_name=cf
```

The above script can also be used to run ROME and MEMIT from the same file. We have a common underlying code-base for calculating the key and value vectors. The update equations for ROME, MEMIT and EMMET are in the file unified_editing/unified_main.py 


**Before any experiment is run**, there might be need to update ```sys.path.append('/path/to/unified-model-editing')``` in the files 'experiments/evaluate_unified_editing.py' and 'experiments/py/eval_utils_zsre.py' 

## Downstream Evaluation

**downstream_tasks** specifies the downstream tasks to run. Available tasks: nli,rte,mrpc,sentiment_analysis,dialogue,nli,cola,sst,hellaswag

**number_of_few_shots** is the number of few shots for each downstream task. Specify the number of few shots for each task, separated by commas. number_of_few_shots must be same length as downstream_tasks. Its default value is 0 when the flag is not provided

**number_of_tests** is the number of tests for all downstream tasks. The default to using the entire test dataset if the flag is not provided

Example:
To run nli, sst and mmlu with 2,3,3 few shots respectively, run the following command:

```python
python experiments/evaluate_unified_editing.py \
--alg_name=EMMET \
--num_edits=4 \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--ds_name=cf \
--do_downstream_eval=True \
--downstream_eval_steps=20 \
--downstream_tasks=nli,sst,mmlu \
--number_of_few_shots=2,3,3 \
--number_of_tests=20
```

## How to Cite
If you find our work useful, please cite it using the following:

```bibtex
@article{gupta2024unified,
  title={A Unified Framework for Model Editing},
  author={Gupta, Akshat and Sajnani, Dev and Anumanchipalli, Gopala},
  journal={arXiv preprint arXiv:2403.14236},
  year={2024}
}
```

```bibtex
@article{gupta2024model,
  title={Model Editing at Scale leads to Gradual and Catastrophic Forgetting},
  author={Gupta, Akshat and Rao, Anurag and Anumanchipalli, Gopala},
  journal={arXiv preprint arXiv:2401.07453},
  year={2024}
}
```
