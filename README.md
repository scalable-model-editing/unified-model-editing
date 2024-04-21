# EMMET is not able to perform batched edits for batch sizes of 10k matching the performance of MEMIT
This repo supports model editing with Llama-3

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
