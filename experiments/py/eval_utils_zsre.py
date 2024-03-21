"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

#sys.path.append('/path/to/unified-model-editing')
from eval_utils_counterfact import *

from dsets import AttributeSnippets


def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
    lm_prompt = False,
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


    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    generation_prompts = record["generation_prompts"]

    if not lm_prompt:
        rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    else:
        rewrite_prompts = [record["requested_rewrite"]["prompt"]]
        generation_prompts.append(record["requested_rewrite"]["prompt"])

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"])["input_ids"]
    #print(target_tok)
    if 'llama' in model.config._name_or_path.lower():
        target_tok = target_tok[2:]

    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = [
        el + tok.decode(target_tok[:i]) if 'llama' not in model.config._name_or_path.lower() or i ==0 else el + ' ' + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]

    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    # Predict for neighborhood prompts exactly same as above
    neighborhood_target = neighborhood_prompts['target']
    neighborhood_prompt = neighborhood_prompts['prompt']

    target_tok_n = tok(" " + neighborhood_target)["input_ids"]
    #print(target_tok_n)
    if 'llama' in model.config._name_or_path.lower():
        target_tok_n = target_tok_n[2:]

    inp_prompts_og_n = [neighborhood_prompt]
    inp_prompts_n = [
        el + tok.decode(target_tok_n[:i]) if 'llama' not in model.config._name_or_path.lower() or i ==0 else el + ' ' + tok.decode(target_tok_n[:i])
        for el in inp_prompts_og_n
        for i in range(len(target_tok_n))
    ]

    inp_targets_n = [
        tok.decode(target_tok_n[i])
        for _ in range(len(inp_prompts_og_n))
        for i in range(len(target_tok_n))
    ]

    neighborhood_correct = test_batch_prediction_acc(model, tok, inp_prompts_n, inp_targets_n)



    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    gen_texts = generate_fast(
        model,
        tok,
        generation_prompts,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ret["text"] = gen_texts

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    #print(prompts)

    #print(target)

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        #print(correct_id)

        # Temporary hack to deal with foreign characters.
        if 'llama' in model.config._name_or_path.lower():
            correct_id = correct_id[:, 1].squeeze()
        else:
            correct_id = correct_id[:, 0].squeeze()##this is the original code

        #print(correct_id)
        #print(ans)
        #print((ans == correct_id).detach().cpu().numpy().tolist())
        #print(tok.decode(ans))

        prediction_text = [tok.decode(token).strip().lower() for token in ans]
        original_text = [token.strip().lower() for token in target]



        text_comparison = []
        for i in range(min(len(prediction_text), len(original_text)) ):
            text_comparison.append(prediction_text[i] == original_text[i])

        #print()
        #print(prediction_text)
        #print(original_text)
        #print(text_comparison)
        
        if 'llama' in model.config._name_or_path.lower():
            return text_comparison
        else:
            return (ans == correct_id).detach().cpu().numpy().tolist()
