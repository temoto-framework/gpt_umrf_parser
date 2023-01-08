import os
from itertools import permutations

import torch
from torch.utils.data import DataLoader, random_split

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

    """
    This function creates permutations of all UMRF examples provided
    by the training set, i.e., self.input_information
    """

    def create_k_many_prompt(self, all_prompts: list, k: int) -> list:
        k_permutation_examples = permutations(all_prompts, k)

        k_permutations_str = []
        for item in k_permutation_examples:
            k_permutations_str.append(''.join(item))
        return k_permutations_str

    def create_robert_prompt(self, new_instruction: str) -> list:
        prompt_template_list = []

        preamble = "Extract data from natural language setences (NL_SENTENCE) and store them " \
            "in JSON format (DESIRED_JSON). I will provide you examples of the desired JSON structure.\n"

        nl_prefix = "NL_SENTENCE:"
        desired_json_prefix = "DESIRED_JSON:"

        suffix = nl_prefix + "\'" + new_instruction + "\'\n" + desired_json_prefix

        robert_prompts = []
        for batch_idx, (nl_instruction, visual_info, umrf_graph) in enumerate(self.input_information):
            part_1 = nl_prefix + "\'" + visual_info + ' ' + nl_instruction + "\'\n"
            part_2 = desired_json_prefix + umrf_graph + "\n"
            robert_prompts.append(part_1 + part_2)

        # Create k-combinatorial examples
        k_combination_examples = self.create_k_many_prompt(robert_prompts, 2)

        # Synthesizes the preamble and suffix strings to each combination example
        prompt_template_list = []
        for item in k_combination_examples:
            prompt_template_list.append(preamble + item + suffix)
        return prompt_template_list

    """
    This function returns a list of lists. The first dimensions is proportional
    to the number of validation examples (times the number of methods used).
    The second dimension is the number of permutations of k-prompts.
    """

    def create_prompts(self) -> list:
        prompt_template_list = []

        # # TODO:
        # # Method 1: Naively concatenate all input information
        # for batch_idx, (nl_instruction, visual_info, umrf_graph) in enumerate(self.input_information):
        #     prompt_template_list.append(nl_instruction + ' ' + visual_info + ' ' + umrf_graph + ' ' + new_instruction)

        # Method 2: Use Robert's Prompt Template
        for batch_idx, (nl_instruction, visual_info, umrf_graph) in enumerate(self.validation_exs):
            prompt_template_list.append(
                self.create_robert_prompt(nl_instruction))
        return prompt_template_list


if __name__ == '__main__':
    umrf_data_path = os.getcwd() + '/umrf_data/*'
    full_umrf_dataset = UMRF(umrf_data_path)

    training_exs, validation_exs = random_split(full_umrf_dataset, [20, 7],
                                                generator=torch.Generator().manual_seed(42))

    prompts = Prompt(input_information=training_exs,
                     validation_exs=validation_exs)
