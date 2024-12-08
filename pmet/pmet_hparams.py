from dataclasses import dataclass
from typing import List
from typing_extensions import Literal


from util.hparams import HyperParams
import yaml

@dataclass
class PMETHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float
    nll_loss_factor: float
    # Module templates
    rewrite_module_tmp: str
    rewrite_module_tmps: List[str]
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False

    # @classmethod
    # def from_hparams(cls, hparams_name_or_path: str):

    #     if '.yaml' not in hparams_name_or_path:
    #         hparams_name_or_path = hparams_name_or_path + '.yaml'

    #     with open(hparams_name_or_path, "r") as stream:
    #         config = yaml.safe_load(stream)
    #         config = super().construct_float_from_scientific_notation(config)

    #     assert (config and config['alg_name'] == 'PMET') or print(f'PMETHyperParams can not load from {hparams_name_or_path}, '
    #                                             f'alg_name is {config["alg_name"]} ')
    #     return cls(**config)