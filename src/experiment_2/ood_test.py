from argparse import ArgumentParser
import sys
import yaml

from datasets import Dataset
import evaluate
import openai
from openai import OpenAI
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

sys.path.append("/vast/home/slwanna/git_repos/gpt_umrf_parser_cleanup/gpt_umrf_parser/")

from src.icl_builder import Prompt
from src.helper import get_instruction_template


# Below function is necessary for OpenAI API queries
def completions_with_backoff(**kwargs):
    model_name = kwargs["model"]
    if "gpt-3.5-turbo-instruct" in model_name:
        return client.completions.create(**kwargs)
    elif "gpt-4" in model_name:
        return client.chat.completions.create(**kwargs)


if __name__ == '__main__':

    print("Beginning Out-of-Distrinbution (OOD) Testing Script.")

    # Step 0. Load in system and experiment parameters
    parser = ArgumentParser()
    parser.add_argument("--config", default="None", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as f:
        sys_config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    with open(sys_config["experiment_config_path"]) as f:
        exp_config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    model_id = exp_config["model_id"]

    # Step 1a. Load in in-distribution examples
    id_umrf_df = pd.read_csv(sys_config["dataset_dir_path"] + exp_config["id_dataset_file_name"])

    # Step 1b. Split full dataset into train and test datasets.
    id_umrf_train_ds = Dataset.from_pandas(id_umrf_df[:exp_config["num_train_exs"]])
    test_set_df = pd.read_csv(sys_config["dataset_dir_path"] + exp_config["ood_dataset_file_name"])

    # Step 2. Create ICL prompts with QA pairs (final ds is test_set_df)
    prompts_df = Prompt(id_umrf_train_ds).create_prompts()
    task_instruction_template_type = exp_config["task_instruction_type"]
    instruction = get_instruction_template(task_instruction_template_type)

    prompt_series = [prompts_df["prompts"]] * len(test_set_df)
    test_set_df = test_set_df.assign(prompts=pd.Series(prompt_series).values)
    test_set_df = test_set_df.explode("prompts")

    test_set_df["fully_assembled_prompts"] = (test_set_df["prompts"]
                                              + "### Natural Language Instruction: \n"
                                              + test_set_df["nl_instruction"]
                                              + '\n' +
                                              "### JSON format: \n")

    # Step 3. Set up LLM inference
    if "gpt" in model_id:
        # Begin using OpenAI API
        openai.api_key = os.getenv("GPT_API_KEY")
        client = OpenAI()
    else:
        # Begin using HuggingFace Transformer local models
        tokenizer = AutoTokenizer.from_pretrained(sys_config["model_dir_path"] + exp_config["model_id"])
        model = AutoModelForCausalLM.from_pretrained(sys_config["model_dir_path"] + exp_config["model_id"],
                                                    load_in_8bit=True,
                                                    device_map="auto")
        
        model.eval()
        gen_config = GenerationConfig(
            max_new_tokens=1024,
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
            generation_config=gen_config
        )
        pipe.tokenizer.pad_token_id = model.config.eos_token_id

    for i in range(3):
        sampled_df = test_set_df.sample(n=100, random_state=i)
        test_ds_final = Dataset.from_pandas(sampled_df.replace(np.NAN, ''))

        output_dictionary = pd.DataFrame(data={
            "example_number": [],  
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
            "prediction": [],
        }
        )

        OUTPUT_FILE_PATH = (sys_config["output_dir_path"] + 
                            exp_config["exp_dir"] +
                            exp_config["model_id"] +
                            "_" + exp_config["task_instruction_type"] +
                            "_" + "ood_distribution" +
                            "_trial_" + str(i) +
                            "_output_generations_" +
                            exp_config["ood_dataset_file_name"])

        output_dictionary.to_csv(OUTPUT_FILE_PATH)

        for item in test_ds_final:
            if "gpt-3.5-turbo-instruct" in exp_config["model_id"]:
                prediction = completions_with_backoff(
                    model="gpt-3.5-turbo-instruct",
                    prompt=[item["fully_assembled_prompts"]],
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    temperature=0,
                )
                prediction = prediction.choices[0].text
            elif "gpt-4" in exp_config["model_id"]:
                prediction = completions_with_backoff(
                    model="gpt-4",
                    messages=[{"role": "user", "content": item["fully_assembled_prompts"]},],
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    temperature=0,
                )
                prediction = prediction.choices[0].message.content
            else:
                prediction = pipe(item["fully_assembled_prompts"])
                prediction = prediction[0]["generated_text"]
            df = {
                "example_number": [item["example_number"]],
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
                "prediction": [prediction],
            }
            temp_df = pd.DataFrame(data=df)
            temp_df.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, index=True)
   