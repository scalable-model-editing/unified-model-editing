from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .GRACE import GRACE
from .grace_hparams import GRACEHyperParams
from .utils import tokenize
from util import nethook


def apply_grace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: GRACEHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    assert len(request) > 1, "GRACE does not allow batch editing"
    """
    {'case_id': 15746, 'prompt': '{} was born in', 'relation_id': 'P19', 'target_new': {'str': 'Calgary', 'id': 'Q36312'}, 'target_true': {'str': 'Adelaide', 'id': 'Q5112'}, 'subject': 'Tony Vidmar'}
    """ 
    prompt = request['prompt'].format(request['subject'])
    target_new = request['target_new']['str']
    ground_truth = request['target_true']['str']

    request = {'prompt': prompt,
        'target_new': target_new,
        'ground_truth': ground_truth,
        'portability': {},
        'locality': {}
        }
        
    if copy:
        model = deepcopy(model)
    device = torch.device(f'cuda:{hparams.device}')
    editor = GRACE(model=model, config=hparams, device=device)
    tokens = tokenize(request, tokenizer=tok, device=device)
    editor.edit(config=hparams, tokens=tokens,edit_id=request['target_new'])
            
    weights_copy = editor.reset_layer


    return editor.model, (weights_copy, {})

