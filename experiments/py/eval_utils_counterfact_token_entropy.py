"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.new_generate import generate_fast
from util.perplexity import perplexity
#from experiments.py.generate_paraphrases import *

def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )

    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]

    if gpt_paraphrase:
        original_phrase = [record["requested_rewrite"]["prompt"].format(subject) + " " + target_new["str"]]
        #paraphrase_prompts = generate_prompts(original_phrase[0], subject, target_new["str"], record)
        if not paraphrase_prompts:
            paraphrase_prompts = record["paraphrase_prompts"]
    else:
        paraphrase_prompts = record["paraphrase_prompts"]

    neighborhood_prompts = record["neighborhood_prompts"]
    generation_prompts = record["generation_prompts"]
    attribute_prompts = record["attribute_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
        attribute_prompts,
    ]

    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
        [0 for _ in range(len(attribute_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
        llama,
    )


    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]

        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"

        #text_generation generates text for generation_prompts for fluency calculation
        gen_stats = text_generation(
            model,
            tok,
            generation_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )

        ret.update(gen_stats)

    return ret
        