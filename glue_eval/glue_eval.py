import sys
import json

sys.path.append('/home/anuragrao/model-editing')
from glue_eval.sst_eval import SSTEval
from glue_eval.mrpc_eval import MRPCEval
from glue_eval.cola_eval import COLAEval
from glue_eval.rte_eval import RTEEval

class GLUEEval():
    def __init__(self, model, tokenizer):
        self.sst_eval = SSTEval(model, tokenizer)
        self.mrpc_eval = MRPCEval(model, tokenizer)
        self.cola_eval = COLAEval(model, tokenizer)
        self.rte_eval = RTEEval(model, tokenizer)

    def _save_generations(self, record_path, generations, task):
        #store individual generation file
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)

    def evaluate(self, glue_results, record_path, sst_flag = False, mrpc_flag = False, cola_flag = False, rte_flag = False, gen_len = 3):
        if sst_flag:
            result_dict, generations = self.sst_eval.evaluate(gen_len)
            glue_results['sst'] = result_dict
            self._save_generations(record_path, generations, 'sst')

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

        return glue_results


        

