from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_fscore_support
from glue_eval.useful_functions import load_data 
import time

class RTEEval():
    def __init__(self, model, tokenizer, eval_split = 'validation'):
        self.model = model
        self.tokenizer = tokenizer

        #initialize dataset
        #dataset = load_dataset("glue", "rte")
        #self.eval_dataset = dataset[eval_split]
        self.eval_dataset = load_data('glue_eval/dataset/rte.pkl') 

        self._initialize_prompts()


    def _initialize_prompts(self):
        self.few_shot_context = '''\
Cyrus captured Babylon without a battle, and remedied the evils done by previous Assyrian and Babylonian rulers by sending prisoners in Babylonia back to their original homelands and aiding in the restoration of temples of the gods of various nations.
question: Babylon surrendered to Cyrus without going to battle. True or False?
answer: False

Successful plaintiffs recovered punitive damages in Texas discrimination cases 53% of the time.
question: Legal costs to recover punitive damages are a deductible business expense. True or False?
answer: True

The gastric bypass operation, also known as stomach stapling, has become the most common surgical procedure for treating obesity.
question: Obesity is medically treated. True or False?
answer: False

'''
        self.prefix_prompt = ''
        self.postfix_prompt = 'answer:' 

    def _create_prompt(self, example):
        prompt = example['sentence1'] + '\n'
        prompt += 'question: ' + example['sentence2'] + ' True or False?'  + '\n'

        input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

        return input_prompt


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('answer:')[-1].strip().strip()

        if 'False' in answer_text:
            return 0
        elif 'True' in answer_text:
            return 1

        return -1


    def evaluate(self, gen_len = 3, print_logs = False):
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
        for s, element in enumerate(self.eval_dataset):    
            sentence1 = element['sentence1']
            sentence2 = element['sentence2']
            label = element['label']

            input_prompt = self._create_prompt(element)
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
                'sentence1': sentence1,
                'sentence2': sentence2,
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

        print(result_dict)

        return result_dict, stored_generations

if __name__ == '__main__':
    '''dataset = load_dataset("glue", "rte")
    eval_dataset = dataset['train']

    count = 0
    for example in eval_dataset:
        print(example)
        print()

    exit()'''


    # Load the tokenizer and model
    model_name = 'EleutherAI/gpt-j-6b'
    #model_name = 'gpt2-xl'
    #model_name = '/data/akshat/lingua-models/Llama-2-7b-hf'
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    rte_eval = RTEEval(model, tokenizer)
    rte_eval.evaluate(print_logs='True')