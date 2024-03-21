from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data
import math
import time

class COLAEval():
    def __init__(self, model, tokenizer, eval_split = 'validation'):
        self.model = model
        self.tokenizer = tokenizer

        #initialize dataset
        #dataset = load_dataset("glue", "cola")
        #self.eval_dataset = dataset[eval_split]
        self.eval_dataset = load_data('glue_eval/dataset/cola.pkl')

        self._initialize_prompts()


    def _initialize_prompts(self):
        self.few_shot_context = '''\
Is this sentence linguistically acceptable?
Sentence : Bill pushed Harry off the sofa for hours.
Answer : No

Is this sentence linguistically acceptable?
Sentence : Bill floated down the river for hours.
Answer : Yes

Is this sentence linguistically acceptable?
Sentence : It is important for the more you eat, the more careful you to be.
Answer : No

Is this sentence linguistically acceptable?
Sentence : It is important for you to be more careful, the more you eat.
Answer : Yes

Is this sentence linguistically acceptable?
Sentence : Mary will believe Susan, and you will Bob.
Answer : Yes

Is this sentence linguistically acceptable?
Sentence : You will Bob believe.
Answer : No

'''
        self.prefix_prompt = 'Is this sentence linguistically acceptable?\n'
        self.postfix_prompt = 'Answer :' 

    def _create_prompt(self, example):
        prompt = 'Sentence : ' + example['sentence'] + '\n'

        input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

        return input_prompt, example['sentence'], example['label']


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('Answer :')[-1].strip().strip()

        if 'Yes' in answer_text:
            return 1
        elif 'No' in answer_text:
            return 0

        return -1

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