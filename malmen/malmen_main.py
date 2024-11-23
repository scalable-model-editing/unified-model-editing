# import os
# from copy import deepcopy
# from typing import Dict, List, Any, Tuple

# import hydra
# import torch
# from collections import deque
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from util.globals import *

# from . import MALMEN
# from .malmen_hparams import MALMENHyperParams

# class MalmenRewriteExecutor:
#     def __init__(self):
#         self.is_init = False

#     def init_model(self, model, tok, params: MALMENHyperParams):

#         assert params.archive is not None or print(f'Training weights Needed....')
#         # Customize the gpt2xl and tokenizer
#         self.model = model
#         self.tokenizer = tok
#         # add_padding(self.tokenizer, self.model)

#         # Load the trained MEND model
#         self.alg = MALMEN(self.model, params, lambda: deepcopy(self.model))
#         d = torch.load(params.archive, map_location=f'cuda:{params.device}')
#         self.alg.load_state_dict(d["model"])
#         if params.model_parallel:
#             self.alg.net.to(deque(self.alg.model.parameters(), maxlen=1)[0].device)
#         else:
#             self.alg.to(torch.device(f'cuda:{params.device}'))


#     def reset_model(self):
#         self.is_init = False
#         del self.model, self.tokenizer, self.alg

#     def apply_to_model(
#         self,
#         model: AutoModelForCausalLM,
#         tok: AutoTokenizer,
#         requests: List[Dict],
#         hparams: MALMENHyperParams,
#         copy=False,
#         return_orig_weights=False,
#     ):
#         """
#         Given a request, for example
#         {'prompt': '{} has the position of',
#          'subject': 'Charles Herman Helmsing',
#          'relation_id': 'P39',
#          'target_new': {'str': 'President', 'id': 'Q11696'},
#          'target_true': {'str': 'bishop', 'id': 'Q29182'}}
#         Returns a dictionary of numpy arrays that specifies
#         how mend will change the weights of the model.
#         """

#         if not self.is_init:
#             self.init_model(model, tok, hparams)

#         weights_copy = {}
#         model = deepcopy(self.model) if copy else self.model
#         assert len(requests) >= hparams.n_edits, "The number of requests must be greater than or equal to the value of n_edits."
#         # Define i/o
#         requests = requests[:hparams.n_edits]
#         batchs = []
#         for i in range(hparams.n_edits // hparams.batch_size):
#             batch = requests[i * hparams.batch_size : (i+1)*hparams.batch_size]
#             targets = [
#                 (" " if request["target_new"][0] != " " else "")
#                 + request["target_new"]
#                 for request in batch
#             ]
#             sentences = [
#                 request["prompt"] + targets[i]
#                 for i, request in enumerate(batch)
#             ]

#             # Tokenize
#             sent_tok = self.tokenizer(sentences, padding=True, return_tensors="pt").to(
#                 f"cuda:{hparams.device}"
#             )
#             target_tok = self.tokenizer(targets, padding=True, return_tensors="pt").to(
#                 f"cuda:{hparams.device}"
#             )

#             # Define labels
#             label_tok = deepcopy(sent_tok["input_ids"])
#             for i in range(label_tok.size(0)):
#                 target_len = target_tok["attention_mask"][i].sum()
#                 padding_len = (
#                     sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
#                 )
#                 label_tok[i][: -target_len - padding_len] = -100
#                 label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100

#             edit_inner = dict(
#                 input_ids=sent_tok["input_ids"],
#                 attention_mask=sent_tok["attention_mask"],
#                 labels=target_tok['input_ids'],
#             )

#             batchs.append(edit_inner)
#         # Run M
#         module_kv_map = self.alg.cache(batchs)
#         param_shifts = self.alg.predict_param_shifts(module_kv_map)
#         with torch.no_grad():
#             for n, p in self.model.named_parameters():
#                 if n in hparams.inner_params:
#                     if return_orig_weights and n not in weights_copy:
#                         weights_copy[n] = p.detach().clone()
#         self.alg.edit_model(param_shifts, False)

#         return self.alg.model, (weights_copy, {})
 

from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf
import importlib
from hydra import initialize, compose
from .data.base import make_edit_loader, make_loader
from .data.zsre import ZSREDataset
from .malmen_hparams import MALMENHyperParams
from .editor.malmen import MALMEN
from .model import make_model

# @hydra.main(version_base=None, config_path="config", config_name="config")
# def main(config: DictConfig):    
#     model = make_model(config.model).to(config.model_device)

#     editor_module = importlib.import_module(f"editor.{config.editor.name}")
#     editor_class = getattr(editor_module, config.editor.name.upper())
#     editor = editor_class(config, model) #already trained

def apply_malmen_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MALMENHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
        
    request = requests[0]
    assert len(request) > 1, "GRACE does not allow batch editing"
    """
    {'case_id': 15746, 'prompt': '{} was born in', 'relation_id': 'P19', 'target_new': {'str': 'Calgary', 'id': 'Q36312'}, 'target_true': {'str': 'Adelaide', 'id': 'Q5112'}, 'subject': 'Tony Vidmar'}
    """ 

    with initialize(config_path="config", version_base=None):
            config = compose(config_name="config")

    print("malmen_main.py line 46")
    data_class = ZSREDataset
    train_loader = make_edit_loader(config, data_class, tok, requests)

    print("malmen_main.py line 49")
    editor_class = MALMEN
    editor = editor_class(config, model)
    editor.model_editing(train_loader)

    print("malmen_main.py line 55")
    return editor.model, ("weights_copy", {})
