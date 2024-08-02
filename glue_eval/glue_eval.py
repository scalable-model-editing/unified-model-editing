import sys
import json

#sys.path.append('/home/anuragrao/model-editing')
sys.path.append('/data/christinefang/unified-model-editing')

from glue_eval.sst_eval import SSTEval
from glue_eval.mrpc_eval import MRPCEval
from glue_eval.cola_eval import COLAEval
from glue_eval.rte_eval import RTEEval
from glue_eval.mmlu_llama_eval import MMLUEval

from glue_eval.cross_entropy_loss import Cross_entropy


class GLUEEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0):
        self.sst_eval = SSTEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = number_of_few_shots)
        self.mrpc_eval = MRPCEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = number_of_few_shots)
        self.cola_eval = COLAEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = number_of_few_shots)
        self.rte_eval = RTEEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = number_of_few_shots)
        self.mmlu_eval = RTEEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = number_of_few_shots)
        self.validation_loss = Cross_entropy(model, tokenizer)


    def _save_generations(self, record_path, generations, task):
        #store individual generation file
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)



    def evaluate(self, glue_results, record_path, llama = False, cross_entropy = False, sst_flag = False, mmlu_flag = False, mrpc_flag = False, cola_flag = False, rte_flag = False, gen_len = 3):
        if cross_entropy:
            loss = self.validation_loss.evaluate()
            glue_results['cross_entropy_loss'] = loss

        if sst_flag:
            result_dict, generations = self.sst_eval.evaluate(gen_len, llama = llama)
            glue_results['sst'] = result_dict
            self._save_generations(record_path, generations, 'sst')

        if mmlu_flag:
            result_dict, generations = self.mmlu_eval.evaluate(gen_len, llama = llama)
            glue_results['mmmlu'] = result_dict
            self._save_generations(record_path, generations, 'mmlu')

        if mrpc_flag:
            result_dict, generations = self.mrpc_eval.evaluate(gen_len, llama = llama)
            glue_results['mrpc'] = result_dict
            self._save_generations(record_path, generations, 'mrpc')

        if cola_flag:
            result_dict, generations = self.cola_eval.evaluate(gen_len, llama = llama)
            glue_results['cola'] = result_dict
            self._save_generations(record_path, generations, 'cola')

        if rte_flag:
            result_dict, generations = self.rte_eval.evaluate(gen_len, llama = llama)
            glue_results['rte'] = result_dict
            self._save_generations(record_path, generations, 'rte')
            
        return glue_results


        

