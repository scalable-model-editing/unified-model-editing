from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data
import math
import torch
import time
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 50

class MMLUEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/mmlu.pkl', number_of_few_shots)
        self.eval_dataset = self.eval_dataset[:number_of_tests] if not (number_of_tests is None) else self.eval_dataset

        self._initialize_prompts()



    def _initialize_prompts(self):
        self.glue_prompt = "Question: "
        self.postfix_prompt = 'True or False? answer: '
        self.few_shot_context = ""
        for _, few_shot in enumerate(self.few_shots):
            prompta = '(A) ' + few_shot['choices'][0] + '\n'
            promptb = '(B) ' + few_shot['choices'][1] + '\n'
            promptc = '(C) ' + few_shot['choices'][2] + '\n'
            promptd = '(D) ' + few_shot['choices'][3] + '\n'
            self.few_shot_context += f'Question: {few_shot['question']}\n{prompta + promptb + promptc + promptd}Answer: {self._get_label(few_shot['answer'])}\n'
        print("FEWWWW_SHOTTT")
        print(self.few_show_context)

    def _get_label(self, example_label):
        if example_label == 0:
            return 'A'
        if example_label == 1:
            return 'B'
        if example_label == 2:
            return 'C'
        if example_label == 3:
            return 'D'
    
    def _create_prompt(self, example):
        prompt = 'Question: ' + example['question'] + '\n'
        prompta = '(A) ' + example['choices'][0] + '\n'
        promptb = '(B) ' + example['choices'][1] + '\n'
        promptc = '(C) ' + example['choices'][2] + '\n'
        promptd = '(D) ' + example['choices'][3] + '\n'

        input_prompt = self.few_shot_context + prompt + prompta + promptb + promptc + promptd + 'Answer:'
        return input_prompt, example['question'], example['answer']

    def _get_answer(self, text):
        if 'A\n' in text:
            return 0
        elif 'B\n' in text:
            return 1
        elif 'C\n' in text:
            return 2
        elif 'D\n' in text:
            return 3
        return -1

    def evaluate(self, gen_len = 10, llama = False, print_logs = False):

        a_tok, b_tok, c_tok, d_tok = (self.tokenizer(f" {n}\n")["input_ids"] for n in ['A', 'B', 'C', 'D'])

        if llama:#'llama-2' in model.config._name_or_path.lower():
            print("LLAMMA TRUE")
            a_tok = a_tok[2:]
            b_tok = b_tok[2:]
            c_tok = c_tok[2:]
            d_tok = d_tok[2:]

        a_len, b_len, c_len, d_len = (len(n) for n in [a_tok, b_tok, c_tok, d_tok])

        correct = 0
        incorrect = 0
        invalid = 0

        correct_new = 0
        incorrect_new = 0

        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []
        new_equal_old = []

        start = time.time()
        for s, example in enumerate(self.eval_dataset):
            input_prompt, sentence, label = self._create_prompt(example)
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

                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits    #the model takes in a list of prompts. logits = a x b x c where a is the number of prompts. Then bxc is the output logits. 

                if llama:
                    logits = logits[:, 1:, :]

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
            answer = self._get_answer(generated_text.replace(input_prompt_text, ''))

            prob_a = np.exp(-probs[0])
            prob_b = np.exp(-probs[1])
            prob_c = np.exp(-probs[2])
            prob_d = np.exp(-probs[3])

            gen_texts1 = gen_texts[0]


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

            if answer == -1:
                invalid += 1
            else:

                if answer == label:
                    correct += 1
                else:
                    incorrect += 1

            if new_answer == label:
                correct_new += 1
            else:
                incorrect_new += 1

            exp_temp_dict = {
                'sentence': sentence,
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
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'correct_new': correct_new,
            'incorrect_new': incorrect_new,
            'total': s+1,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end-start,
        }
        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = "/data/akshat/models/gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    mmlu_eval = MMLUEval(model, tokenizer)
    result_dict, stored_generations = mmlu_eval.evaluate(print_logs='True'