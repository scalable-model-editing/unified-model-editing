from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split
import time
import torch
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 50
## IMPORTANT, few shot learning is important as it allow the answer coming out from the model to be formatted. 

class DIALOGUE_Eval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/dialogue.pkl', number_of_few_shots) 
        self.eval_dataset = self.eval_dataset[:number_of_tests] if not (number_of_tests is None) else self.eval_dataset

        self._initialize_prompts()

    def _initialize_prompts(self): 
        self.postfix_prompt = "The answer should be exact A or B or C or D. Among A through D, the answer is"
        self.few_shot_context = ""
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context += f"Q: Given the following: {few_shot['article']}\n Which choice is correct? Answer Chioces:\n (A){few_shot["options"][0]} \n (B){few_shot["options"][1]} \n (C){few_shot["options"][2]} \n (D){few_shot["options"][0]} \n {self.postfix_prompt} {few_shot['answers']}"
    
    def _create_prompt(self, example):
        input_prompt =   f"Q: Given the following: {example['article']}\n Which choice is correct? Answer Chioces:\n (A){example["options"][0]} \n (B){example["options"][1]} \n (C){example["options"][2]} \n (D){example["options"][0]} \n {self.postfix_prompt}"
        return input_prompt, example['options'], example['article'], self._get_label(example['answers'])
    
    def _get_answer(self, generated_text):
        answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()

        if 'A' == answer_text:
            return 0
        if 'B' == answer_text:
            return 1
        if 'C' == answer_text:
            return 2
        if 'D' == answer_text:
            return 3
        return -1

    def _get_label(self, example_label):
        if 'A' == example_label:
            return 0
        if 'B' == example_label:
            return 1
        if 'C' == example_label:
            return 2
        if 'D' == example_label:
            return 3
        
    def evaluate(self, gen_len = 3, print_logs = False):
        a_tok, b_tok, c_tok, d_tok = (self.tokenizer(f" {n}\n")["input_ids"][-2:] for n in ['A', 'B', 'C', 'D'])
        a_len, b_len, c_len, d_len = (len(n) for n in [a_tok, b_tok, c_tok, d_tok])

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
            input_prompt, options, article, label = self._create_prompt(example)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])
            print(prefix_tok_len)

            probs = [0, 0, 0, 0]
            gen_texts = [0, 0, 0, 0]
            gen_texts2 = [0, 0, 0, 0]
            dic = {0: [a_tok, a_len], 1: [b_tok, b_len], 2: [c_tok, c_len], 3: [d_tok, d_len]}
            
            max_len = input_prompt_ids.shape[1] + gen_len
            suffixes = ['A', 'B', 'C', 'D']
            for i in range(4):
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i]}\n"], return_tensors="pt").to('cuda')
                logits = self.model(**prompt_tok).logits.to('cuda')

                cur_len = dic[i][1]

                for j in range(cur_len):
                    cur_tok = dic[i][0][j]
                    probs[i] += -torch.nn.functional.log_softmax(
                    logits[0, prefix_tok_len + j - 1, :], dim=0
                    )[cur_tok].item()
                probs[i] /= cur_len

                gen_texts[i] = self.tokenizer.decode(logits[0, prefix_tok_len - 1 : prefix_tok_len + cur_len - 1, :].argmax(dim = -1))

            output = self.model.generate(input_prompt_ids,max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self._get_answer(generated_text)
            prob_a = np.exp(-probs[0])
            prob_b = np.exp(-probs[1])  
            prob_c = np.exp(-probs[2])  
            prob_d = np.exp(-probs[3]) 
            predictions.append(answer)
            labels.append(label)
            def hi(prob_a, prob_b, prob_c, prob_d):
                if prob_a > max(prob_b, prob_c, prob_d):
                    return 0
                elif prob_b > max(prob_a, prob_c, prob_d):
                    return 1
                elif prob_c > max(prob_b, prob_a, prob_d):
                    return 2
                elif prob_d > max(prob_b, prob_c, prob_a):
                    return 3
                return -1
            
            new_answer = hi(prob_a, prob_b, prob_c, prob_d)
            predictions_new.append(new_answer)
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
                'options': options,
                'article': article,
                'label': label,
                'input_prompt': input_prompt_text,
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'invalid': True if answer == -1 else False,
                'correct': answer == label,
                'prob_a': prob_a,
                'prob_b': prob_b,
                'prob_c': prob_c,
                'prob_d': prob_d,
                'answer_new': hi(prob_a, prob_b, prob_c, prob_d),
                'correct_new': new_answer == label,
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
    model_name = '/data/akshat/lingua-models/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    dialogue_eval = DIALOGUE_Eval(model, tokenizer)
    dialogue_eval.evaluate(print_logs='True')