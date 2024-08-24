from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import math
import torch
import time
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 100

class GMS8K_Eval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/gms8k.pkl', number_of_few_shots, number_of_tests)
        self._initialize_prompts()

    def _initialize_prompts(self):
        self.postfix_prompt = 'Solve it step by step, before your leave your numeric answer, have 4 hashtags (i.e. ####) before it.'
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"{few_shot["question"]} {self.postfix_prompt}\n {few_shot['answer']}\n")

    def _create_prompt(self, example, gen_len):
        question = f"{example["question"]} {self.postfix_prompt}\n"
        question_token_length = len(self.tokenizer(question)["input_ids"])
        remaining_token_length = MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP[self.model.config._name_or_path.lower().split('/')[-1]] - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 0:
                break 
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        return input_prompt
    
    def _get_answer(self, generated_text):
        answer_text = generated_text.split('####')[-1].strip().strip()
        return int(answer_text)

    def evaluate(self, gen_len = 10, print_logs = False):
        correct = 0
        incorrect = 0
        invalid = 0

        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0
        
        predictions = []
        labels = []
        stored_generations = []
        start = time.time()
        for s, example in enumerate(self.eval_dataset):
            input_prompt, sentence, label = self._create_prompt(example)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            max_len = input_prompt_ids.shape[1] + gen_len

            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            answer = self._get_answer(generated_text)
            predictions.append(answer)
            labels.append(label)
            if answer == -1:
                invalid += 1
            else:

                if answer == label:
                    correct += 1

                    if label == 1:
                        pos_correct += 1
                    elif label == 0:
                        neg_correct += 1

                else:
                    incorrect += 1

                    if label == 1:
                        pos_incorrect += 1
                    elif label == 0:
                        neg_incorrect += 1

            exp_temp_dict = {
                'sentence': sentence,
                'label': label,
                'input_prompt': input_prompt_text,
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'invalid': True if answer == -1 else False
            }
            stored_generations.append(exp_temp_dict)

            if print_logs:
                mcc = matthews_corrcoef(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                print(generated_text)
                print(correct, incorrect, invalid, s+1, '|', pos_correct, neg_correct, '|', pos_incorrect, neg_incorrect, '|ACC: ', correct / (correct + incorrect + invalid), '|MCC:', mcc, '|F1:', f1)
                print('--'*50)

        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'f1': f1,
            'mcc': mcc,
            'time': end-start,
        }
        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = 'gpt2-xl'
    #model_name = 'EleutherAI/gpt-j-6b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    cola_eval = COLAEval(model, tokenizer)
    result_dict, stored_generations = cola_eval.evaluate(print_logs='True')