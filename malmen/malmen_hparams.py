from dataclasses import dataclass
from typing import List
from util.hparams import HyperParams

@dataclass
class MALMENHyperParams(HyperParams):
    name_or_path: str
    class_name: str
    model_name: str
    edit_modules: List[str]
    half: bool
