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
from util.generate import generate_fast
from util.perplexity import perplexity


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
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")


    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])

    if 'llama-2' in model.config._name_or_path.lower():
        a_tok = a_tok[2:]
        b_tok = b_tok[2:]
        prefix_lens = [lengths -1 for lengths in prefix_lens]

    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    with torch.no_grad():
        logits = model(**prompt_tok).logits

    if 'llama-2' in model.config._name_or_path.lower():
        logits = logits[:, 1:, :]

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []
    gen_text_list = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)
            gen_text = tok.decode(logits[i, prefix_lens[i // 2] - 1 : prefix_lens[i // 2] + cur_len-1, :].argmax(dim = -1))
            gen_text_list.append(gen_text)

    prompts = [(prefix, suffix) for prefix in prefixes for suffix in [target_new, target_true]]

    prompt_gen = generate_fast(
            model,
            tok,
            prefixes,
            n_gen_per_prompt=1,
            top_k=1,
            max_out_len=50,
            )
    text_sliding = []
    correct_sliding = []

    for i in range(len(prompts)):
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
        # check if text generated at gen_texts[i//2] contains prompts[i][1] (suffix)
            correct_sufx = prompts[i][1] # is a string
            prefx = prompts[i][0]

            #post edit correctness
            full_text = prompt_gen[i//2] # is a string
            generated_suffix = full_text[len(prefx):]
            text_sliding.append(generated_suffix)
            correct = correct_sufx in generated_suffix
            correct_sliding.append(correct)

    post_list = [{"target_new_prob": probs[i].item(),
                "target_true_prob": probs[i + 1].item(),
                "target_new": target_new,
                "target_true":  target_true,
                'prompt': prefixes[e] ,
                'generated_text': gen_text_list[e],
                'correct': targets_correct[e],
                'sliding_text': text_sliding[e],
                'sliding_correct': correct_sliding[e],
                }
        for e, i in enumerate(range(0, len(probs), 2))]

    return post_list, correct_sliding #targets_correct

    text_sliding = []
    correct_sliding = []

    post_list = [{"target_new": probs[i].item(), 
                "target_true": probs[i + 1].item(), 
                'prompt': prefixes[e] , 
                'generated_text': gen_text_list[e], 
                'correct': targets_correct[e]}
        for e, i in enumerate(range(0, len(probs), 2))]

    return post_list, correct_sliding


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    #remove input prompt when calculating entropy
    entropy_texts = [gen_texts[i][len(prefixes[i]):] for i in range(len(gen_texts))]

    ngram_entropy = n_gram_entropy(entropy_texts)
    tok_entropy = token_entropy(entropy_texts, tok)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "adjusted_token_entropy": tok_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})
    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()

def token_entropy(gen_texts, tokenizer, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_token_entropy(txt, tokenizer) for txt in gen_texts]
    ).item()

def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, False,  n, None)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


### ADJUSTED NGRAM 
def compute_token_entropy(sentence, tokenizer, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]

    entropy_list = []
    for n in ns:
        ## Calculate the number of ngrams (either tokens or words) for normalizing
        tokens = tokenizer(sentence)['input_ids']
        ngrams = nltk.ngrams(tokens, n)
        ngram_count = len(list(ngrams))

        fdist = compute_freq(sentence, True,  n, tokenizer)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()
        if ngram_count == 1:
            entropy_list.append(0.0)
        else:
            entropy_list.append((np.sum(-freqs * np.log(freqs) / np.log(2)))/(np.log(ngram_count)/np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, token_entropy,  n=2, tokenizer=None):
    if token_entropy:
        tokens = tokenizer(sentence)['input_ids']
    else:
        tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
