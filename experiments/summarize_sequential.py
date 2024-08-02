import collections
import json
from pprint import pprint
from typing import List, Optional
import sys

import numpy as np
from scipy.stats import hmean

sys.path.append('/data/christinefang/unified-model-editing')
from util.globals import *


def gen_metrics(
    dir_name = None,
    runs: Optional[List] = None,
    first_n_cases=None,
    abs_path=False,
    get_uncompressed=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
    #for run_dir in [Path(abs_path)]:
        print(f"rundir: {str(run_dir)}")

        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("*case_*.json"))

        files.sort(key=lambda x: int(str(x).split("_")[-3]))
        file_wise_results = {}

        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")

            num_edits = int(str(case_file).split('/')[-1].split('_')[1])

            # Probability metrics for which new should be lower (better) than true
            for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs", "attribute_prompts_probs"]:
                if key not in data["post"]:
                    continue

                sum_key_discrete = f"post_{key.split('_')[0]}_success"
                sum_key_cont = f"post_{key.split('_')[0]}_diff"
                cur_sum[sum_key_discrete].append(
                    np.mean(
                        [
                            x["post_target_true_prob"] > x["post_target_new_prob"]
                            for x in data['post'][key]
                        ]
                    )
                )
                cur_sum[sum_key_cont].append(
                    np.mean(
                        [
                            np.exp(-x["post_target_new_prob"]) - np.exp(-x["post_target_true_prob"])
                            for x in data['post'][key]
                        ]
                    )
                )

                sum_key_discrete = f"pre_{key.split('_')[0]}_success"
                sum_key_cont = f"pre_{key.split('_')[0]}_diff"

                cur_sum[sum_key_discrete].append(
                    np.mean(
                        [
                            x["target_true_prob"] > x["target_new_prob"]
                            for x in data['post'][key]
                        ]
                    )
                )
                cur_sum[sum_key_cont].append(
                    np.mean(
                        [
                            np.exp(-x["target_new_prob"]) - np.exp(-x["target_true_prob"])
                            for x in data['post'][key]
                        ]
                    )
                )

            # Probability metrics for which true should be lower (better) than new
            sum_key_discrete = f"post_neighborhood_success"
            sum_key_cont = f"post_neighborhood_diff"
            key = "neighborhood_prompts_probs"
            if key in data['post']:
                cur_sum[sum_key_discrete].append(
                    np.mean(
                        [
                            x["post_target_true_prob"] < x["post_target_new_prob"]
                            for x in data['post'][key]
                        ]
                    )
                )
                cur_sum[sum_key_cont].append(
                    np.mean(
                        [
                            np.exp(-x["post_target_true_prob"]) - np.exp(-x["post_target_new_prob"])
                            for x in data['post'][key]
                        ]
                    )
                )

            """
            sum_key_discrete = f"pre_neighborhood_success"
            sum_key_cont = f"pre_neighborhood_diff"
            key = "neighborhood_prompts_probs"
            if key in data['post']:
                cur_sum[sum_key_discrete].append(
                    np.mean(
                        [
                            x["target_true_prob"] < x["target_new_prob"]
                            for x in data['post'][key]
                        ]
                    )
                )
                cur_sum[sum_key_cont].append(
                    np.mean(
                        [
                            np.exp(-x["target_true_prob"]) - np.exp(-x["target_new_prob"])
                            for x in data['post'][key]
                        ]
                    )
                )
            """

            # Accuracy-based evaluation metrics
            for i in ["rewrite", "paraphrase", "neighborhood"]:#, "attribute"]:
                sum_key = f"post_{i}_acc"
                key = f"post_{i}_prompts_correct"

                cur_sum[sum_key].append(np.mean(data['post'][key]))

                sum_key = f"pre_{i}_acc"
                key = f"{i}_prompts_correct"

                cur_sum[sum_key].append(np.mean(data['post'][key]))

            # Generation metrics that can be directly averaged
            for key in ["ngram_entropy", "reference_score"]:
                if key in data['post']:
                    cur_sum[f"pre_{key}"].append(data['post'][key])

            if "post_ngram_entropy" in data['post']:
                cur_sum[f"post_ngram_entropy"].append(data['post']['post_ngram_entropy'])


    data_filename = 'metrics/llama_run_047_emmet.json'
    with open(data_filename, "wt") as f:
            json.dump(cur_sum, f, indent=2)

    pprint(cur_sum)
    #return uncompressed if get_uncompressed else summaries
    return cur_sum


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data."