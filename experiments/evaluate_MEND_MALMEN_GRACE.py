import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import torch
import torch.nn as nn

import sys
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('/data/christinefang/unified-model-editing')
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from malmen import MALMENHyperParams, apply_malmen_to_model
from grace import GRACEHyperParams, apply_grace_to_model
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from emmet import EMMETHyperParams, apply_emmet_to_model
from util import nethook
from util.globals import *

from glue_eval.glue_eval import GLUEEval

ALG_DICT = {
    "EMMET": (EMMETHyperParams, apply_emmet_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "MALMEN": (MALMENHyperParams, apply_malmen_to_model),
    "GRACE": (GRACEHyperParams, apply_grace_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    args,
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    sequential: bool,
    downstream_eval_steps: int,
    downstream_tasks: str,
    number_of_few_shots: str,
    number_of_tests: int,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )

    try:
        hparams = params_class.from_json(params_path)
    except:
        params_path = HPARAMS_DIR / alg_name / hparams_fname
        hparams = params_class.from_json(params_path)

    if not (run_dir / "params.json").exists():
        #shutil.copyfile(params_path, run_dir / "params.json")
        ####ADDING HPARAMETERS TO SAVE
        hparams_to_save = hparams.__dict__
        hparams_to_save['model_name'] = model_name
        hparams_to_save['algo_name'] = alg_name
        hparams_to_save['dataset'] = ds_name
        hparams_to_save['n_edits'] = num_edits
        hparams_to_save['sequential'] = sequential
        hparams_to_save['sample_num'] = args.sample_num
        
        with open(run_dir / "params.json", "w") as f:
            json.dump(hparams_to_save, f, indent=1)

            
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")

        # model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        # tok = AutoTokenizer.from_pretrained(model_name)
        # tok.pad_token = tok.eos_token
        
        local_models = {"Llama-2-7b": "/data/akshat/models/Llama-2-7b-hf", "gpt2-xl":  "/data/akshat/models/gpt2-xl",  "EleutherAI/gpt-j-6B":  "/data/akshat/models/gpt-j-6b"}
        # model = AutoModelForCausalLM.from_pretrained(local_models[model_name]).cuda()
        model = AutoModelForCausalLM.from_pretrained(local_models[model_name])
        model = nn.DataParallel(model)
        # Move the model to the GPU
        model = model.cuda()
        tok = AutoTokenizer.from_pretrained(local_models[model_name])
        tok.pad_token = tok.eos_token


    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    #if num_edits > 1:
    #    assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")


    # Iterate through dataset
    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)

    #load indices file and initialize dataset class
    if ds_name in 'cf':
        indices_filename = 'data/counterfact_sampled_unique_cf_10_20000.json'
        dataset = CounterFactDataset('data')
    if ds_name in 'mcf':
        indices_filename = 'data/counterfact_sampled_unique_mcf_10_20000.json'
        dataset = CounterFactDataset('data')
    elif ds_name == 'zsre':
        indices_filename = 'data/zsre_sampled_unique_10_10000.json'
        dataset = MENDQADataset('data', tok)

    f = open(indices_filename)
    sampled_indices = json.load(f)


    # Iterate through dataset
    for r, e in enumerate(range(0, len(sampled_indices[args.sample_num]), num_edits)):
        record_chunks = []
        for element_index in sampled_indices[args.sample_num][e: min(e+num_edits, len(sampled_indices[args.sample_num]))]:
            datapoint = dataset.__getitem__(element_index)
            record_chunks.append(datapoint)

        if r == 4000:
            break

        case_result_template = str(run_dir / "{}_{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, r, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()


                ##decode the number of few shots
        if args.do_downstream_eval and downstream_tasks is None:
            raise ValueError("No downstream tasks were provided.")

        if args.do_downstream_eval:
            downstream_tasks_list = downstream_tasks.split(",")

            number_of_few_shots_str_list = number_of_few_shots.split(",")

            number_of_few_shots_dict = {}

            if len(number_of_few_shots_str_list) == 1 and number_of_few_shots_str_list[0] == "-1":

            ## this is the default case

            ## if the user didn't specify the number of few shots, then it will be defualt to be the string of -1

            ## in that case, we shoudl assune that all of the few shots per tasks is zero

                number_of_few_shots_list = [0 for _ in range(len(downstream_tasks_list))]

            else:

                assert len(number_of_few_shots_str_list) == len(downstream_tasks_list), f"Error, if you have {len(downstream_tasks_list)} number of downstream tasks, you should also specify that many few shot examples for each downstream tasks, but we received only {len(number_of_few_shots_str_list)} of few shot examples assigned"

                number_of_few_shots_list = []

                for item in number_of_few_shots_str_list:

                    try:

                        converted_item = int(item)

                        number_of_few_shots_list.append(converted_item)

                    except ValueError:

                        raise ValueError(f"Error: '{item}' cannot be converted to an integer. the few shot example number must be an integer")

            for i, downstream in enumerate(downstream_tasks_list):

                number_of_few_shots_dict[downstream + "_number_of_few_shots"] = number_of_few_shots_list[i]


        if r == 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
            glue_results = {'edit_num': -1}
            # weight_norms = get_weight_norms(model)
            # glue_results['weight_norms'] = weight_norms

            out_file = glue_save_location + "base.json"
            if (args.sequential or num_edits > 1) and args.do_downstream_eval:
                glue_eval = GLUEEval(model, tok, number_of_tests, **number_of_few_shots_dict)
                flags = [_ in downstream_tasks for _ in ['sst', 'mmlu', 'mrpc', 'cola', 'rte', 'nli', 'dialogue', 'hellaswag']]
                glue_results = glue_eval.evaluate(glue_results, out_file, False, *flags)

            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)


        gen_test_vars = [snips, vec]        
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, r, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue


            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "post": ds_eval_method(
                    model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),
                )
            }
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)
        
        print("about to apply algo")
        start = time()
        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]} if alg_name != "MALMEN" else  {"case_id": record["case_id"], **record}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )

        exec_time = time() - start
        print("Execution took", exec_time)


        # weight_norms = get_weight_norms(model)
        glue_results = {
            'edit_num': r,
            'case_id': case_ids,
            # 'weight_norms': weight_norms,

            }

        out_file = glue_save_location + "{}_case_{}.json".format(r, record["case_id"])#stores the last case ID of the batch
        if (args.sequential or num_edits > 1) and args.do_downstream_eval and (r + 1) % args.downstream_eval_steps == 0:
            glue_eval = GLUEEval(edited_model, tok, number_of_tests, **number_of_few_shots_dict)
            flags = [_ in downstream_tasks for _ in ['sst', 'mmlu', 'mrpc', 'cola', 'rte', 'nli', 'dialogue', 'hellaswag']]
            glue_results = glue_eval.evaluate(glue_results, out_file, False, *flags)
        

        #store the individual overall result file
        output_filename = out_file.replace('.json', '_glue.json')
        with open(output_filename, "w") as f:
            json.dump(glue_results, f, indent=4)


        # Evaluate new model
        start = time()
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, r, record["case_id"]))

            with open(out_file, 'r+') as f:
                data = json.load(f)
                data['time'] = exec_time,

                post_list = ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),
                )
                for key, value in post_list.items():
                    if key in ["paraphrase_prompts_probs", "neighborhood_prompts_probs", "rewrite_prompts_probs", "attribute_prompts_probs"]:
                        for i in range(len(data['post'][key])):
                            data['post'][key][i]['post_sliding_text'] = post_list[key][i]['sliding_text']
                            data['post'][key][i]['post_sliding_correct'] = post_list[key][i]['sliding_correct']
                            data['post'][key][i]['post_target_new_prob'] = post_list[key][i]['target_new_prob']
                            data['post'][key][i]['post_target_true_prob'] = post_list[key][i]['target_true_prob']
                    if key in ["rewrite_prompts_correct", "paraphrase_prompts_correct", "neighborhood_prompts_correct", "attribute_prompts_correct" "text", "ngram_entropy", "essence_score"]:
                        data['post']['post_' + key] = post_list[key]

                f.seek(0)        # <--- should reset file position to the beginning.
                json.dump(data, f, indent=4)
                f.truncate()


        # if args.model_save_interval > 0 and (r + 1) % args.model_save_interval == 0:
        #     model_save_folder = args.model_save_location + alg_name + '/' + f"run_{str(run_id).zfill(3)}" + '/edits_' + str(e + 1)
        #     os.makedirs(model_save_folder)
        #     model.save_pretrained(model_save_folder)


        if not sequential:
            # Restore original weights
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)
        
def get_weight_norms(model):
    layers = [
        "transformer.h.45.mlp.c_proj.weight",
        "transformer.h.45.mlp.c_fc.weight",
        "transformer.h.46.mlp.c_proj.weight",
        "transformer.h.46.mlp.c_fc.weight",
        "transformer.h.47.mlp.c_proj.weight",
        "transformer.h.47.mlp.c_fc.weight"
        ]

    weight_norms = {
        layer: torch.norm(nethook.get_parameter(model, layer)).detach().cpu().item()
        for layer in layers
    }

    return weight_norms



def get_model_distance(original_weights, model_new, model_hpar):
    state_dict_original = original_weights
    state_dict_new = model_new.state_dict()

    distances_dict = {}
    for layer in model_hpar.layers:
        if isinstance(layer, str) and 'transformer' in layer:
            rewrite_layer = layer
        else:
            rewrite_layer = model_hpar.rewrite_module_tmp.format(str(layer)) + '.weight'

        distance = torch.norm(state_dict_original[rewrite_layer] - state_dict_new[rewrite_layer].cpu()) / state_dict_original[rewrite_layer].numel()
        distances_dict[layer] = distance.detach().cpu().item()
    
    return distances_dict


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_num",
        type=str,   
        default="0",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--alg_name",
        choices=["MEND", "MALMEN", "GRACE"],
        default="MEND",
        help="Editing algorithm to use. Results are savn results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "Llama-2-7b"],
        default="gpt2-xl",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--sequential",
        type=bool,
        default=True,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=5,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--downstream_tasks",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--number_of_few_shots",
        type=str,
        default="-1",
        required=False,
    )
    parser.add_argument(
        "--number_of_tests",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--do_downstream_eval",
        type=bool,
        default=False,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--model_save_interval",
        type=int,
        default=10,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--model_save_location",
        type=str,
        default='/data/christinefang/unified-model-editing/edited_models/',
        help="END LOCATION WITH /",
    )


    parser.set_defaults(skip_generation_tests=False, conserve_memory=True)
    args = parser.parse_args()

    main(
        args,
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        args.sequential,
        args.downstream_eval_steps,
        args.downstream_tasks,
        args.number_of_few_shots,
        args.number_of_tests,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )