import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util.globals import *

from .layer_stats import layer_stats
from .layer_stats_dynamic import layer_stats_dynamic
from .rome_hparams import ROMEHyperParams

# Cache variables
inv_mom2_cache = {}


def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: ROMEHyperParams,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    constraint_rome=False,
    constraint_lambda = 100,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache
    if hparams.dynamic:##makes sure C is recomputed each iteration. clear cache
        inv_mom2_cache = {}

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )

        if hparams.dynamic:
            stat, key_vectors, value_vectors = layer_stats_dynamic(
                model,
                tok,
                layer_name,
                STATS_DIR,
                mom2_dataset,
                to_collect=["mom2"],
                sample_size=mom2_n_samples,
                precision=mom2_dtype,
                hparams=hparams
            )
            inv_mom2_cache['key_vectors'] = key_vectors
            inv_mom2_cache['value_vectors'] = value_vectors

        else:
            stat, _ = layer_stats(
                model,
                tok,
                layer_name,
                STATS_DIR,
                mom2_dataset,
                to_collect=["mom2"],
                sample_size=mom2_n_samples,
                precision=mom2_dtype,
            )


        if constraint_rome:
            print('\n\n\nCALCULATING constraint ROME update\n\n\n')
            C = stat.mom2.moment()
            I = torch.eye(C.shape[0])

            C_regularized = C + constraint_lambda * I
            inv_mom2_cache[key] = torch.inverse(
                C_regularized.to("cuda")
            ).float()  # Cast back to float32

        else:
            inv_mom2_cache[key] = torch.inverse(
                stat.mom2.moment().to("cuda")
            ).float()  # Cast back to float32

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    ####u is same as k* in the the derivation

    # Apply inverse second moment adjustment
    u = cur_repr

    #bias correction part
    if hparams.bias_update:
        bias_correction = torch.tensor([1], dtype=u.dtype, device=u.device)
        u = torch.cat((u, bias_correction), dim = 0)

    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            tok,
            hparams,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        ) @ u.unsqueeze(1)
        u = u.squeeze()

    #This operation is same as C^-1 k*^T in paper. Then they normalize it, which is not done in the paper.

    return u / u.norm(), inv_mom2_cache
