import os
import timeit

import torch
from torch.utils.data import DataLoader, random_split
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from umrf_dataset import UMRF
from prompt import Prompt


"""
This class will be responsible for defining reward metrics and
different search methods resembling those outlined in (Jiang2020c)
to find optimal discrete prompt
"""


class JiangPrompt:
    def __init__(self, prompt_templates: list, validation_exs: DataLoader, device: str):
        self.prompt_templates = prompt_templates
        self.validation_exs = validation_exs

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)


    def top_one_selection(self):
        # Step 0. Grab validation labels for UMRF graphs
        valid_ex_ground_truths = []
        for _, (_, _, umrf_graph) in enumerate(self.validation_exs):
            valid_ex_ground_truths.append(umrf_graph)

        h = len(self.prompt_templates)
        w = len(self.prompt_templates[0])
        
        accuracies = torch.zeros(h, w).cpu()
        i = 0
        for valid_ex in self.prompt_templates:
            print(i)
            j = 0
            label = valid_ex_ground_truths[i]
            # max seq input size = 4096 for gpt-3 OpenAI
            # max seq input size = 1024 for gpt-2
            for prompt in valid_ex:
                prompt_tokenized = self.tokenizer(
                    prompt, truncation=True, max_length=512, return_tensors="pt").to(device)

                # Step 1. run every prompt through gpt-2
                outputs = self.model.generate(**prompt_tokenized, return_dict_in_generate=True,
                                            output_scores=True, max_length=1024)
                output_tokens = self.tokenizer.decode(outputs['sequences'][0])

                # # Step 2. check accuracy on ea. prompt against validation label
                accuracies[i, j] = self.acc(label, output_tokens)

                j = j + 1
            i = i + 1
        # Step 3. choose highest acc. score
        # average acc across validation examples
        avg_accs = torch.mean(accuracies, dim=0)
        # then choose prompt with highest acc
        best_prompt_indx = torch.argmax(avg_accs)
        # choose best prompt construction
        # NOTE: when reporting in paper, remove the validation_ex
        # from the prompt!! (we just want the generic template
        # despite arbirarily grabbing this format from the first
        # validation example)
        best_prompt = self.prompt_templates[0][best_prompt_indx]

        return best_prompt


    """
    Calculates the character-level accuracy for a single
    UMRF ground-truth against a single UMRF model decoding
    """

    def acc(self, input_information: str, model_output: str):
        ground_truth = input_information[2].lower()
        model_output = model_output.lower()

        penalize_extra_decodings = len(model_output) - len(ground_truth)

        num_correct_characters = 0
        i = 0
        if penalize_extra_decodings >= 0:
            for char in ground_truth:
                if char == model_output[i]:
                    num_correct_characters = num_correct_characters + 1
                i = i + 1
            acc = num_correct_characters / \
                (len(ground_truth) + abs(penalize_extra_decodings))

        else:
            for char in model_output:
                if char == ground_truth[i]:
                    num_correct_characters = num_correct_characters + 1
                i = i + 1
            acc = num_correct_characters / \
                (len(model_output) + abs(penalize_extra_decodings))

        return acc


def check_for_gpu() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(230)
        # Empty GPU Cache
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    return device


if __name__ == '__main__':
    # set seed for reproducible experiments
    torch.manual_seed(230)

    device = check_for_gpu()

    umrf_data_path = os.getcwd() + '/umrf_data/*'
    umrf_dataset = UMRF(umrf_data_path)

    training_exs, validation_exs = random_split(umrf_dataset, [20, 7],
                                                generator=torch.Generator().manual_seed(42))

    prompts = Prompt(input_information=training_exs,
                     validation_exs=validation_exs)
    prompts_list = prompts.create_prompts()

    jiang_opt = JiangPrompt(prompts_list, validation_exs, device)
    print(jiang_opt.top_one_selection())
