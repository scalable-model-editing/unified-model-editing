from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split
import time
import torch
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 50
## IMPORTANT, few shot learning is important as it allow the answer coming out from the model to be formatted. 

class NLIEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/nli.pkl', number_of_few_shots) 
        self.eval_dataset = self.eval_dataset[:number_of_tests] if not (number_of_tests is None) else self.eval_dataset

        self._initialize_prompts()
    def _initialize_prompts(self):
        self.postfix_prompt = 'True or False? answer:' 
        self.few_shot_context = ""
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context += f'{few_shot["sentence1"]} entails the {few_shot["sentence2"]} {self.postfix_prompt} {("True" if few_shot["label"] == "entailment" else "False")}\n'  
    
    def _create_prompt(self, example):
        input_prompt = self.few_shot_context + f'{example["sentence1"]} entails the {example["sentence2"]} {self.postfix_prompt}'
        return input_prompt, example['sentence1'], example['sentence2'], self._get_label(example['label'])
    
    def _get_answer(self, generated_text):
        answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()

        if 'True' in answer_text:
            return 1
        elif 'False' in answer_text:
            return 0
        return -1

    def _get_label(self, example_label):
        if 'entailment' == example_label:
            return 1
        return 0

    def evaluate(self, gen_len = 3, print_logs = False):
        true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['True', 'False'])

        if 'llama-2' in self.model.config._name_or_path.lower():
            true_tok = true_tok[2:]
            false_tok = false_tok[2:]

        true_len, false_len = (len(n) for n in [true_tok, false_tok])
        print(true_len)
        suffixes = {0: ['True', true_tok, true_len], 1: ['False', false_tok, false_len]}

        correct = 0
        incorrect = 0
        invalid = 0

        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []

        start = time.time()
        for s, example in enumerate(self.eval_dataset):
            input_prompt, sentence1, sentence2, label = self._create_prompt(example)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])

            if 'llama-2' in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            max_len = input_prompt_ids.shape[1] + gen_len
            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self._get_answer(generated_text)

            predictions.append(answer)
            labels.append(label)

            #### EVALUATE NEW ACC 
            
            probs = [0 for _ in suffixes.keys()]
            gen_texts = [0 for _ in suffixes.keys()]

            for i in range(len(suffixes.keys())):
                print(suffixes[i][0])
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i][0]}"], return_tensors="pt").to('cuda')

                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits

                if 'llama-2' in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]

                cur_len = suffixes[i][2]

                for j in range(cur_len):
                    cur_tok = suffixes[i][1][j]
                    probs[i] += -torch.nn.functional.log_softmax(
                    logits[0, prefix_tok_len + j - 1, :], dim=0
                    )[cur_tok].item()
                probs[i] /= cur_len
                gen_texts[i] = self.tokenizer.decode(logits[0, prefix_tok_len - 1 : prefix_tok_len + cur_len - 1, :].argmax(dim = -1))

            prob_true = np.exp(-probs[0])
            prob_false = np.exp(-probs[1])

            print(f"prob_true: {prob_true}, prob_false: {prob_false}")

            answer_new = 1 if prob_true > prob_false else 0
            predictions_new.append(answer_new)
            print(f"prediction: {answer}, true: {label}")
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
                'sentence1': sentence1,
                'sentence2': sentence2,
                'input_prompt': input_prompt_text,
                'true_answer': 'True' if label == 1 else 'False', 
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'correct': answer == label,
                'prob_true': prob_true,
                'prob_false': prob_false,
                'answer_new': 'True' if answer_new == 1 else 'False', 
                'correct_new': answer_new == label,
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
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end-start,
        }

        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = '/data/akshat/lingua-models/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    nli_eval = NLIEval(model, tokenizer)
    nli_eval.evaluate(print_logs='True')