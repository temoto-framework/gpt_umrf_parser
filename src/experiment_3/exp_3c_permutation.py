from argparse import ArgumentParser
import yaml

from datasets import Dataset
import evaluate
import openai
from openai import OpenAI
import os
import numpy as np
import pandas as pd
import sys
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          GenerationConfig,
                          pipeline)

sys.path.append("/vast/home/slwanna/git_repos/gpt_umrf_parser_cleanup/gpt_umrf_parser/")

from src.icl_builder import Prompt
from src.helper import get_instruction_template


# Below function is necessary for OpenAI API queries
def completions_with_backoff(**kwargs):
    return client.completions.create(**kwargs)


if __name__ == '__main__':
    # Load in experimental parameters
    parser = ArgumentParser()
    parser.add_argument("--config", default="None", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as f:
        sys_config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    with open(sys_config["experiment_config_path"]) as f:
        exp_config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    """Number of Few-Shot Dataset Creation"""

    # Step 1. Set up LLM inference
    model_id = exp_config["model_id"]

    bleu_fn = evaluate.load("bleu")  # NLG metric to use

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
            return_full_text=False,
            batch_size=1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            generation_config=gen_config
        )
        pipe.tokenizer.pad_token_id = model.config.eos_token_id


    
    # begin building varied task instructions
    instruction = get_instruction_template(exp_config["task_instruction_type"])

    # Loading in-dist training set.
    ID_DATASET_DIR_PATH = sys_config["dataset_dir_path"] + exp_config["id_dataset_file_name"]
    id_umrf_df = pd.read_csv(ID_DATASET_DIR_PATH)

    # Loading in ood test set
    OOD_DATASET_DIR_PATH = sys_config["dataset_dir_path"] + exp_config["ood_dataset_file_name"]

    # OUTPUT FILE PATHS
    OUTPUT_DIR = sys_config["output_dir_path"] + exp_config["exp_dir"]

    num_few_shots = 3

    for i in range(3):
        # collect 3 icl examples from full id_dataset
        id_dist_icl_exs = id_umrf_df.sample(n=13, random_state=i)
        id_dist_icl_exs_ds = Dataset.from_pandas(id_dist_icl_exs[:exp_config["num_train_exs"]])
        prompts_df = Prompt(id_dist_icl_exs_ds, k=num_few_shots).create_prompts()

        # create id test set
        id_test_df = id_dist_icl_exs[exp_config["num_train_exs"]:].replace(np.NAN, '')
        # create ood test set
        ood_test_df = pd.read_csv(OOD_DATASET_DIR_PATH).replace(np.NAN, '')


        # id test set
        prompt_series = [prompts_df["prompts"]] * len(id_test_df)
        id_test_df = id_test_df.assign(prompts=pd.Series(prompt_series).values)
        id_test_df = id_test_df.explode("prompts")

        id_output_dict = pd.DataFrame(data={
            "file_name": [],  # use of id_test set
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
            "bleu_score": []
        }
        )

        # assemble full prompt
        id_test_df["fully_assembled_prompts"] = (instruction + " \n"
                                                 + id_test_df["prompts"]
                                                 + "### Natural Language Instruction: \n"
                                                 + id_test_df["nl_instruction"]
                                                 + " + " + id_test_df["coords"] + '\n' +
                                                 "### JSON format: \n")

        id_test_ds = Dataset.from_pandas(id_test_df.replace(np.NAN, ''))

        ID_OUTPUT_FILE_PATH = (OUTPUT_DIR + model_id + "_" + exp_config["task_instruction_type"] +
                               "_id_task_trial_" + str(i) + "_k_" + str(num_few_shots) + ".csv")
        id_output_dict.to_csv(ID_OUTPUT_FILE_PATH)

        for item in id_test_ds:
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
                    bleu_score = bleu_fn.compute(predictions=[prediction], references=[[item["graph"]]])
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
                bleu_score = bleu_fn.compute(predictions=[prediction], references=[[item["graph"]]])
            else:
                prediction = pipe(item["fully_assembled_prompts"])
                prediction = prediction[0]["generated_text"]
                bleu_score = bleu_fn.compute(predictions=[prediction],
                                            references=[[item["graph"]]])   
            
            df = {
                "file_name": [item["file_name"]],  # use of id_test set
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
                "prediction": [prediction[0].split("### Natural Language Instruction", 1)[0]],
                "bleu_score": [bleu_score["bleu"]]
            }
            temp_df = pd.DataFrame(data=df)
            temp_df.to_csv(ID_OUTPUT_FILE_PATH, mode='a', header=False, index=True)

        # ood test set
        prompt_series = [prompts_df["prompts"]] * len(ood_test_df)
        ood_test_df = ood_test_df.assign(prompts=pd.Series(prompt_series).values)
        ood_test_df = ood_test_df.explode("prompts")

        ood_output_dict = pd.DataFrame(data={
            "file_name": [],  # use of id_test set
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
            "bleu_score": []
        }
        )

        ood_test_df["fully_assembled_prompts"] = (instruction + " \n"
                                                 + ood_test_df["prompts"]
                                                 + "### Natural Language Instruction: \n"
                                                 + ood_test_df["nl_instruction"] + '\n' +
                                                 "### JSON format: \n")

        ood_test_ds = Dataset.from_pandas(ood_test_df.replace(np.NAN, ''))

        OOD_OUTPUT_FILE_PATH = (OUTPUT_DIR + model_id + "_" + exp_config["task_instruction_type"] +
                                "_ood_task_ablation_" + str(i) + "_k_" + str(num_few_shots) + ".csv")
        ood_test_df.to_csv(OOD_OUTPUT_FILE_PATH)

        for item in ood_test_ds:
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
                    bleu_score = bleu_fn.compute(predictions=[prediction], references=[[item["graph"]]])
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
                bleu_score = bleu_fn.compute(predictions=[prediction], references=[[item["graph"]]])
            else:
                prediction = pipe(item["fully_assembled_prompts"])
                prediction = prediction[0]["generated_text"]
                bleu_score = bleu_fn.compute(predictions=[prediction],
                                            references=[[item["graph"]]])   
            
            df = {
                "file_name": [item["file_name"]],  # use of id_test set
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
                "bleu_score": [bleu_score["bleu"]]
            }
            temp_df = pd.DataFrame(data=df)
            temp_df.to_csv(ID_OUTPUT_FILE_PATH, mode='a', header=False, index=True)

