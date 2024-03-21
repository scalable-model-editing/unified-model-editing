import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
import random

from util.globals import *

from useful_functions import load_data

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"
        zsre_gen_prompts = load_data(data_dir / 'cf_generation_prompts.pkl')
        print(len(zsre_gen_prompts))

        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        zsre_prompt_filename = data_dir / 'zsre_text_completion_prompts.json'
        with open(zsre_prompt_filename, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)


        data = []
        for i, record in enumerate(raw):

            #create generation prompts
            if i < len(json_object):
                generation_prompts = random.sample(zsre_gen_prompts, 10) + [json_object[i]['text_completion_prompt']]
            else:
                generation_prompts = random.sample(zsre_gen_prompts, 10)


            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"

            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts":{
                            "prompt": record["loc"].replace('nq question:', '').strip() + "?",
                            "target": record["loc_ans"],
                    },
                    "attribute_prompts": [],
                    "generation_prompts": generation_prompts,
                }
            )

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
