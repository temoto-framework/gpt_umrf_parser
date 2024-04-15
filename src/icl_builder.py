"""
All this file does is construct few-shot / ICL examples.
You still need to append the queries.

Generates prompt templates to choose from. Will include permutations along the axes of
+ Number of example UMRF parses
+ Ordering of example UMRF parses
+ Prompt example selections  
"""
import os
from itertools import permutations, combinations
import json
from datasets import Dataset
import pandas as pd


class Prompt:
    def __init__(self, id_ds: Dataset, k=2):
        self.id_input_information = id_ds
        self.k = k

    """
    This function creates permutations of all UMRF examples provided
    by the training set, i.e., self.input_information
    """

    def create_k_many_permutation_prompt(self, all_prompts: list, k: int) -> list:
        k_permutation_examples = permutations(all_prompts, k)

        k_permutations_str = []
        for item in k_permutation_examples:
            k_permutations_str.append(''.join(item))
        return k_permutations_str
    
    def create_k_many_combination_prompt(self, all_prompts: list, k: int)->list:
        k_combination_examples = combinations(all_prompts, k)

        k_combination_examples_str = []
        for item in k_combination_examples:
            k_combination_examples_str.append(''.join(item))
        return k_combination_examples_str

    def create_robert_prompt(self, new_instruction: str) -> list:
        prompt_template_list = []

        preamble = "Extract data from natural language sentences (NL_SENTENCE) and store them " \
            "in JSON format (DESIRED_JSON). I will provide you examples of the desired JSON structure.\n"

        nl_prefix = "NL_SENTENCE:"
        desired_json_prefix = "DESIRED_JSON:"

        suffix = nl_prefix + "\'" + new_instruction + "\'\n" + desired_json_prefix

        robert_prompts = []
        for batch_idx, (nl_instruction, visual_info, umrf_graph) in enumerate(self.input_information):
            part_1 = nl_prefix + "\'" + visual_info + ' ' + nl_instruction + "\'\n"
            part_2 = desired_json_prefix + umrf_graph + "\n"
            robert_prompts.append(part_1 + part_2)

        # Uncomment below line for Experiment 1. (permutation builder)
        # k_permutation_examples = self.create_k_many_permutation_prompt(robert_prompts, self.k)

        # Uncomment below line for Experiment 3. (combination builder)
        k_combination_examples = self.create_k_many_combination_prompt(robert_prompts, self.k)

        # Synthesizes the preamble and suffix strings to each combination example
        prompt_template_list = []
        for item in k_combination_examples:
            prompt_template_list.append(preamble + item + suffix)
        return prompt_template_list

    def create_naive_prompt(self) -> list:
        naive_prompts = []
        length = len(self.id_input_information)
        for batch_idx, ex in enumerate(self.id_input_information):
            umrf_graph = ex["graph"]
            nl_instruction = ex["nl_instruction"]
            visual_info = ' '.join(self.grab_coords(json.loads(ex["graph"])))

            # naive_prompts.append(visual_info + ' + ' +
            #                      nl_instruction + ' + ' + umrf_graph)
            naive_prompts.append("### Natural Language Instruction: \n" + nl_instruction + ' + ' +
                                 visual_info + '\n' + "### JSON format: \n" + umrf_graph)

        # Uncomment below line for Experiment 1. (permutation builder)
        # k_permutation_examples = self.create_k_many_permutation_prompt(naive_prompts, self.k)

        # Uncomment below line for Experiment 3. (combination builder)
        k_combination_examples = self.create_k_many_combination_prompt(naive_prompts, self.k)

        # Synthesizes the preamble and suffix strings to each combination example
        return k_combination_examples

    """
    This function returns a list of lists. The first dimensions is proportional
    to the number of validation examples (times the number of methods used).
    The second dimension is the number of permutations of k-prompts.
    """

    def create_prompts(self) -> pd.DataFrame:
        prompt_template_list = []
        prompt_template_list = self.create_naive_prompt()

        # Method 1: Naively concatenate all input information
        # for batch_idx, ex in enumerate(self.id_input_information):
        #     nl_instruction = ex["nl_instruction"]
        #     prompt_template_list.append(self.create_naive_prompt())

        # # Method 2: Use Robert's Prompt Template
        # for batch_idx, ex in enumerate(self.id_validation_queries):
        #     nl_instruction = ex["nl_instruction"]
        #     prompt_template_list.append(
        #         self.create_robert_prompt(nl_instruction))
        d = {"prompts": prompt_template_list}
        return pd.DataFrame(d)

    def grab_coords(self, umrf_graph) -> list:
        list_of_coords = []

        length_of_actions = len(umrf_graph["umrf_actions"])
        for i in range(length_of_actions):
            action = umrf_graph["umrf_actions"][i]
            if "pose_2d" in action["input_parameters"].keys():
                coords = action["input_parameters"]["pose_2d"]
                x_value = coords["x"]["pvf_value"]
                y_value = coords["y"]["pvf_value"]
                yaw_value = coords ["yaw"]["pvf_value"]
                list_of_coords.append(f"x: {x_value}, y: {y_value}, yaw: {yaw_value}")
        return list_of_coords


if __name__ == '__main__':
    # Use pandas to load in the .csv files containing the
    # in-distribution dataset: ALFRED NL instructions
    # ood complex dataset: for complex (nonsequential) tasks (hierarchical, cyclical)
    # ood industrial dataset: for DOE relevant entities

    UMRF_DATASET_PATH = os.getcwd() + "/datasets/"
    id_umrf_df = pd.read_csv(UMRF_DATASET_PATH + "id_alfred.csv")
    ood_complex_df = pd.read_csv(UMRF_DATASET_PATH + "ood_complex_tasks.csv")
    ood_industrial_df = pd.read_csv(UMRF_DATASET_PATH + "ood_industrial_domain.csv")


    # Use huggingface's Dataset to convert the pandas df to a dataset
    id_umrf_ds = Dataset.from_pandas(id_umrf_df).train_test_split(test_size=0.5, shuffle=False)

    ood_df = pd.concat([ood_complex_df, ood_industrial_df])
    ood_umrf_ds = Dataset.from_pandas(ood_df) # this entire dataset is just test set

    # Create the prompts
    prompts = Prompt(id_ds=id_umrf_ds, ood_ds=ood_umrf_ds)

