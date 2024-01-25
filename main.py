import os

from datasets import Dataset
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

from icl_builder import Prompt
from helper import get_instruction_template

if __name__ == '__main__':
    """
    This section handles all the configs:
    model_type: {llama-7b, codellama-7b, mistral, mixtral, gpt-3}
    task_instruction_template_type: {model_name, rp (roleplay), minimal}
    test_set_id: {id_test_set, ood_domain_test_set, ood_task_type_set}
    """
    PATH_TO_MODEL_WEIGHT_FOLDER = "/home/selmawanna/hf_models/"

    # Model type strings:
    model_string_dict = {"llama-7b": "Llama-2-7b-hf",
                         "codellama": "CodeLlama-7b-Instruct-hf",
                         "mistral": "Mistral-7B-Instruct-v0.2",
                         "mixtral": "Mixtral-8x7B-Instruct-v0.1",  # too large
                         "gpt3": None}  # too large
    model_type = model_string_dict["llama-7b"]

    # Task type strings:
    # "minimal"
    # "rp"
    # or any key from the model_string_dict
    task_instruction_template_type = "minimal"

    # test_set_id types:
    # "id_test"
    # "ood_complex"
    # "ood_industrial"
    test_set_id = "ood_complex"

    """
    This section handles loading in all the ICL example data
    """
    # Step 0. Load in the data
    # Use pandas to load in the .csv files containing the
    # in-distribution dataset: ALFRED NL instructions
    # ood complex dataset: for complex (nonsequential) tasks (hierarchical, cyclical)
    # ood industrial dataset: for DOE relevant entities

    UMRF_DATASET_PATH = os.getcwd() + "/datasets/"
    id_umrf_df = pd.read_csv(UMRF_DATASET_PATH + "newest_id_alfred.csv")

    # Step 1. Create id (train/test) and ood test datasets
    # Use huggingface's Dataset to convert the pandas df to a dataset
    id_umrf_train_ds = Dataset.from_pandas(id_umrf_df[:25])

    # Step 2. Create the prompts
    prompts_df = Prompt(id_umrf_train_ds).create_prompts()

    # Step 3. Configure experiment
    instruction = get_instruction_template(task_instruction_template_type)

    test_set_df = None
    if test_set_id == "id_test":
        test_set_df = id_umrf_df[25:]
    elif test_set_id == "ood_complex":
        test_set_df = pd.read_csv(UMRF_DATASET_PATH + "ood_complex_tasks.csv")
    elif test_set_id == "ood_industrial":
        test_set_df = pd.read_csv(UMRF_DATASET_PATH + "ood_industrial_domain.csv")

    # Step 4. Build the massive dataset to query models with
    prompt_series = [prompts_df["prompts"]] * len(test_set_df)
    test_set_df = test_set_df.assign(prompts=pd.Series(prompt_series).values)
    test_set_df = test_set_df.explode("prompts")

    # NOTE: Viz Info only needed for id test set
    # test_set_df["fully_assembled_prompts"] = (test_set_df["prompts"]
    #                                           + "### Natural Language Instruction: \n"
    #                                           + test_set_df["nl_instruction"]
    #                                           + " + " + test_set_df["coords"] + '\n' +
    #                                           "### JSON format: \n")
    test_set_df["fully_assembled_prompts"] = (test_set_df["prompts"]
                                              + "### Natural Language Instruction: \n"
                                              + test_set_df["nl_instruction"]
                                              + '\n' +
                                              "### JSON format: \n")

    # TODO: Change random_state between 1, 2, 3
    # YOU DO NOT HAVE THE TIME ON EARTH TO DO THIS BEFORE THE DEADLINE
    sampled_df = test_set_df.sample(n=150, random_state=1)  # only use on id_test other ood are small enough
    test_ds_final = Dataset.from_pandas(sampled_df.replace(np.NAN, ''))

    # Step 5. Setting up the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL_WEIGHT_FOLDER + model_type)
    model = AutoModelForCausalLM.from_pretrained(PATH_TO_MODEL_WEIGHT_FOLDER + model_type,
                                                 load_in_8bit=True,
                                                 device_map="auto")
    model.eval()

    config = GenerationConfig(
        max_new_tokens=512,
        pad_token_id=model.config.eos_token_id
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        prefix=instruction,
        return_full_text=False,
        batch_size=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        generation_config=config
    )
    pipe.tokenizer.pad_token_id = model.config.eos_token_id

    # , n_graph_levels, human_rated_difficulty, sequential, hierarchical, cyclical
    output_dictionary = pd.DataFrame(data={
        "example_number": [],
        # "file_name": [],  # use of id_test set
        "nl_instruction": [],
        "fully_assembled_prompts": [],
        "n_words_in_n_nl_instruction": [],
        "action_type": [],
        "graph": [],
        "n_graph_nodes": [],
        "n_graph_levels": [],
        "human_rated_difficulty": [],
        "sequential": [],
        "hierarchical": [],
        "cyclical": [],
        "prediction": []
    }
    )

    OUTPUT_FILE_PATH = (os.getcwd() + "/generated_outputs/" + model_type +
                        "_" + task_instruction_template_type +
                        "_" + test_set_id +
                        "_" + "output_generations.csv")

    output_dictionary.to_csv(OUTPUT_FILE_PATH)

    for item in test_ds_final:
        prediction = pipe(item["fully_assembled_prompts"])
        df = {
            "example_number": [item["example_number"]],
            # "file_name": [item["file_name"]],  # use of id_test set
            "nl_instruction": [item["nl_instruction"]],
            "fully_assembled_prompts": [item["fully_assembled_prompts"]],
            "n_words_in_n_nl_instruction": [item["n_words_in_n_nl_instruction"]],
            "action_type": [item["action_type"]],
            "graph": [item["graph"]],
            "n_graph_nodes": [item["n_graph_nodes"]],
            "n_graph_levels": [item["n_graph_levels"]],
            "human_rated_difficulty": [item["human_rated_difficulty"]],
            "sequential": [item["sequential"]],
            "hierarchical": [item["hierarchical"]],
            "cyclical": [item["cyclical"]],
            "prediction": [prediction[0]["generated_text"]]
        }
        temp_df = pd.DataFrame(data=df)
        temp_df.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, index=True)
