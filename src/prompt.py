import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from umrf_dataset import UMRF

"""
Generates prompt templates to choose from. Will include permutations along the axes of
+ Number of example UMRF parses
+ Ordering of example UMRF parses
+ Prompt example selections  
"""
class Prompt:
    def __init__(self, input_information: DataLoader, validation_exs: DataLoader):
        self.input_information = input_information
        self.validation_exs = validation_exs
        
        # for instruction in self.validation_exs:
        #     self.prompt_templates = self.create_prompts(instruction)
        print('hi')
    
    def k_examples(self, k: int):
        raise NotImplementedError


    def create_robert_prompt(self, new_instruction: str) -> str:
        prompt_template_list = []

        preamble = "Extract data from natural language setences (NL_SENTENCE) and store them " \
            "in JSON format (DESIRED_JSON). I will provide you examples of the desired JSON structure.\n"
        
        nl_prefix = "NL_SENTENCE:"
        desired_json_prefix = "DESIRED_JSON:"

        robert_prompt_str = ""
        for batch_idx, (nl_instruction, visual_info, umrf_graph) in enumerate(self.input_information):
            part_1 = nl_prefix + "\'" + visual_info + ' ' + nl_instruction + "\'\n"
            part_2 = desired_json_prefix + umrf_graph + "\n"
            robert_prompt_str = robert_prompt_str + part_1 + part_2

        suffix = nl_prefix + "\'" + new_instruction + "\'\n" + desired_json_prefix

        prompt_template_list.append(preamble + robert_prompt_str + suffix)
        return prompt_template_list


    def create_prompts(self, new_instruction: str) -> list:
        prompt_template_list = []

        # Method 1: Concatenate all input information
        for batch_idx, (nl_instruction, visual_info, umrf_graph) in enumerate(self.input_information):
            prompt_template_list.append(nl_instruction + ' ' + visual_info + ' ' + umrf_graph + ' ' + new_instruction)

        # Method 2: Use Robert's Prompt Template
        prompt_template_list.append(self.create_robert_prompt(new_instruction))
        return prompt_template_list
   
if __name__ == '__main__':
    print('Step 1: load in UMRF dataset')
    umrf_data_path = os.getcwd() + '/umrf_data/*'
    full_umrf_dataset = UMRF(umrf_data_path)

    training_exs, validation_exs = random_split(full_umrf_dataset, [20, 7],
     generator=torch.Generator().manual_seed(42))

    print('Step 2: Create Prompt Obj')
    prompts = Prompt(input_information=training_exs, validation_exs=validation_exs)
    print(prompts.create_robert_prompt(prompts.validation_exs[0][0])[0])


