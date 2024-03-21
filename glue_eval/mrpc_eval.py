from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data
import time

class MRPCEval():
    def __init__(self, model, tokenizer, eval_split = 'validation'):
        self.model = model
        self.tokenizer = tokenizer

        #initialize dataset
        #dataset = load_dataset("glue", "mrpc")
        #self.eval_dataset = dataset[eval_split]
        self.eval_dataset = load_data('glue_eval/dataset/mrpc.pkl') 

        self._initialize_prompts()


    def _initialize_prompts(self):
        #self.few_shot_context = open('prompts/mrpc_1.txt', 'r').read()
        self.few_shot_context = '''\
Are the sentences paraphrases of each other.
Sentence 1: Mr McDevitt has been granted control of three crucial aspects of policing in the Solomons.
Sentence 2: Mr McDevitt has been granted control of three aspects of policing by Commissioner William Morrell.
Answer: No

Are the sentences paraphrases of each other.
Sentence 1: They had published an advertisement on the Internet on June 10, offering the cargo for sale, he added.
Sentence 2: On June 10, the ship's owners had published an advertisement on the Internet, offering the explosives for sale.
Answer: Yes

Are the sentences paraphrases of each other.
Sentence 1: In 2002, Linksys overtook Cisco Systems as the leading wireless equipment vendor, accounting for 14.1 percent of revenue.
Sentence 2: Rolfe said Linksys overtook Cisco Systems last year as the leading supplier of WLAN equipment.
Answer: No

Are the sentences paraphrases of each other.
Sentence 1: The notification was first reported Friday by MSNBC.
Sentence 2: MSNBC.com first reported the CIA request on Friday.
Answer: Yes

Are the sentences paraphrases of each other.
Sentence 1: "The anticipated global sales improvement in the month of June did not materialize", said Chief Financial Officer Robert Rivet.
Sentence 2: "The anticipated global sales improvement in the month of June did not materialize as we had anticipated", the company said.
Answer: Yes

Are the sentences paraphrases of each other.
Sentence 1: That compared with $ 35.18 million, or 24 cents per share, in the year-ago period.
Sentence 2: Earnings were affected by a non-recurring $8 million tax benefit in the year-ago period.
Answer: No

'''
        self.prefix_prompt = 'Are the sentences paraphrases of each other.\n'
        self.postfix_prompt = 'Answer:' 

    def _create_prompt(self, example):
        prompt = 'Sentence 1: ' + example['sentence1'] + '\n'
        prompt += 'Sentence 2: ' + example['sentence2'] + '\n'

        input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

        return input_prompt, example['sentence1'], example['sentence2'], example['label']


    def _get_answer(self, generated_text):
        answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()

        if 'Yes' in answer_text:
            return 1
        elif 'No' in answer_text:
            return 0

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
        for s, example in enumerate(self.eval_dataset):
            input_prompt, sentence1, sentence2, label = self._create_prompt(example)
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

        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    #model_name = 'EleutherAI/gpt-j-6b'
    #model_name = 'gpt2-xl'
    model_name = '/data/akshat/lingua-models/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    mrpc_eval = MRPCEval(model, tokenizer)
    mrpc_eval.evaluate(print_logs='True')