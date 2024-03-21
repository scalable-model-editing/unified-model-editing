import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

from util import nethook
from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

ds = None
STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats_dynamic(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats_dynamic(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=True,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        print('\n'*5)
        print('#'*200)
        print('LOADING DATASET')
        print('#'*200)
        print('\n'*5)

        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = model.config.max_position_embeddings

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    try:
        npos = model.config.n_positions
    except:
        npos = model.config.max_position_embeddings

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    global ds
    if ds == None:
        ds = get_ds()

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    
    total_keys_stored = 0
    dynamic_multiplier = hparams.dynamic_multiplier
    start_time_dynamic = time.time()
    break_outer_loop = False


    key_vectors = []
    value_vectors = []
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):

            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=True, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])

                if hparams.bias_update:#add a bunch of 1's for bias update
                    bias_correction = torch.ones((feats.shape[0], 1), dtype=feats.dtype, device=feats.device)
                    feats = torch.cat((feats, bias_correction), dim=1)
                    value_feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                else:
                    layer_weights = nethook.get_parameter(model, layer_name + ".weight")
                    value_feats = feats @ layer_weights 

                
                #adding dynamic ROME constraint code
                if hparams.dynamic:
                    hidden_dim = feats.shape[1]
                    new_keys = feats.shape[0]

                    #check if keys going over
                    if (total_keys_stored + new_keys) > (hidden_dim * dynamic_multiplier):
                        feats = feats[:(hidden_dim * dynamic_multiplier) - total_keys_stored, :]
                        value_feats = value_feats[:(hidden_dim * dynamic_multiplier) - total_keys_stored, :]
                        new_keys = feats.shape[0]

                    stat.add(feats)
                    total_keys_stored += new_keys
                    key_vectors.append(feats)
                    value_vectors.append(value_feats)

                    print('TOTAL KEYS STORED:', total_keys_stored, 'TIME TAKEN:', time.time()-start_time_dynamic)

                    #finish storing keys when done
                    if total_keys_stored >= (hidden_dim * dynamic_multiplier):
                        break_outer_loop = True
                        break
                ##### END dynamic ROME block
                else:
                    stat.add(feats)

            if break_outer_loop:
                break          
    
    key_vectors = torch.cat(key_vectors, dim = 0)
    value_vectors = torch.cat(value_vectors, dim = 0)

    return (stat, key_vectors, value_vectors)


if __name__ == "__main__":
    main()
