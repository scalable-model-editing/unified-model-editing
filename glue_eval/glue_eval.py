import sys
import json

sys.path.append('/data/christinefang/unified-model-editing')

from glue_eval.sst_eval import SSTEval
from glue_eval.mrpc_eval import MRPCEval
from glue_eval.cola_eval import COLAEval
from glue_eval.rte_eval import RTEEval
from glue_eval.mmlu_eval import MMLUEval
from glue_eval.dialogue_eval import DIALOGUE_Eval
from glue_eval.nli_eval import NLIEval
from glue_eval.hellaswag_eval import HELLASWAG_Eval
from util.perplexity import perplexity
from datasets import load_dataset


class GLUEEval():
    def __init__(self, model, tokenizer, number_of_tests = None, sst_number_of_few_shots = 0, mrpc_number_of_few_shots = 0, cola_number_of_few_shots = 0, rte_number_of_few_shots = 0, mmlu_number_of_few_shots = 0, nli_number_of_few_shots = 0, dialogue_number_of_few_shots = 0, hellaswag_number_of_few_shots = 0):
        self.model = model

        self.tokenizer = tokenizer

        self.sst_eval = SSTEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = sst_number_of_few_shots)

        self.mrpc_eval = MRPCEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = mrpc_number_of_few_shots)

        self.cola_eval = COLAEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = cola_number_of_few_shots)

        self.rte_eval = RTEEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = rte_number_of_few_shots)

        self.mmlu_eval = MMLUEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = mmlu_number_of_few_shots)

        self.nli_eval = NLIEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = nli_number_of_few_shots)

        self.dialogue_eval = DIALOGUE_Eval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = dialogue_number_of_few_shots)

        self.hellaswag_eval = HELLASWAG_Eval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = hellaswag_number_of_few_shots)


    def _save_generations(self, record_path, generations, task):
        #store individual generation file
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)



    def evaluate(self, glue_results, record_path, perplexity_flag = False, sst_flag = False, mmlu_flag = False, mrpc_flag = False, cola_flag = False, rte_flag = False, nli_flag = False, dialogue_flag = False, hellaswag_flag = False, gen_len = 3):
        if perplexity_flag:
            raw_ds = load_dataset(
                        "wikitext",
                        dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")["wikitext"],
                        )
            glue_results['perplexity'] = perplexity(self.model, self.tokenizer, " ".join(raw_ds["train"]['text'][:20]), max_input_length=100)
            
        if sst_flag:
            result_dict, generations = self.sst_eval.evaluate(gen_len)
            glue_results['sst'] = result_dict
            self._save_generations(record_path, generations, 'sst')

        if mmlu_flag:
            result_dict, generations = self.mmlu_eval.evaluate(gen_len)
            glue_results['mmmlu'] = result_dict
            self._save_generations(record_path, generations, 'mmlu')

        if mrpc_flag:
            result_dict, generations = self.mrpc_eval.evaluate(gen_len)
            glue_results['mrpc'] = result_dict
            self._save_generations(record_path, generations, 'mrpc')

        if cola_flag:
            result_dict, generations = self.cola_eval.evaluate(gen_len)
            glue_results['cola'] = result_dict
            self._save_generations(record_path, generations, 'cola')

        if rte_flag:
            result_dict, generations = self.rte_eval.evaluate(gen_len)
            glue_results['rte'] = result_dict
            self._save_generations(record_path, generations, 'rte')

        if nli_flag:
            result_dict, generations = self.nli_eval.evaluate(gen_len)
            glue_results['nli'] = result_dict
            self._save_generations(record_path, generations, 'nli')

        if dialogue_flag:
            result_dict, generations = self.dialogue_eval.evaluate(gen_len)
            glue_results['dialogue'] = result_dict
            self._save_generations(record_path, generations, 'dialogue')

        if hellaswag_flag:
            result_dict, generations = self.hellaswag_eval.evaluate(gen_len)
            glue_results['hellaswag'] = result_dict
            self._save_generations(record_path, generations, 'hellawag')
            
        return glue_results


        

