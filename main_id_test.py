from argparse import ArgumentParser
import yaml

from datasets import Dataset
import evaluate
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

from icl_builder import Prompt
from helper import get_instruction_template

if __name__ == '__main__':
    # Load in experimental parameters
    parser = ArgumentParser()
    parser.add_argument("--config", default="None", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    model_string_dict = config["model_file_names"]
    task_instruction_template_type = config["task_instruction_type"]

    id_umrf_df = pd.read_csv(config["dataset_dir_path"] + config["id_dataset_file_name"])
    id_umrf_train_ds = Dataset.from_pandas(id_umrf_df[:25])

    if config["test_set_id"] == "id_test":
        test_set_df = id_umrf_df[25:]
    else:
        raise NotImplementedError

    prompts_df = Prompt(id_umrf_train_ds).create_prompts()
    instruction = get_instruction_template(task_instruction_template_type)

    prompt_series = [prompts_df["prompts"]] * len(test_set_df)
    test_set_df = test_set_df.assign(prompts=pd.Series(prompt_series).values)
    test_set_df = test_set_df.explode("prompts")

    test_set_df["fully_assembled_prompts"] = (test_set_df["prompts"]
                                              + "### Natural Language Instruction: \n"
                                              + test_set_df["nl_instruction"]
                                              + " + " + test_set_df["coords"] + '\n' +
                                              "### JSON format: \n")

    bleu_fn = evaluate.load("bleu")

    for model_name in model_string_dict.keys():
        tokenizer = AutoTokenizer.from_pretrained(config["model_dir_path"] + config["model_file_names"][model_name])
        model = AutoModelForCausalLM.from_pretrained(config["model_dir_path"] + config["model_file_names"][model_name],
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

            OUTPUT_FILE_PATH = (config["output_dir_path"] + model_name +
                                "_" + config["task_instruction_type"] +
                                "_" + config["test_set_id"] +
                                "_trial_" + str(i) +
                                "_output_generations.csv")

            output_dictionary.to_csv(OUTPUT_FILE_PATH)

            for item in test_ds_final:
                prediction = pipe(item["fully_assembled_prompts"])
                bleu_score = bleu_fn.compute(predictions=[prediction[0]["generated_text"]],
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
                    "prediction": [prediction[0]["generated_text"]],
                    "bleu_score": [bleu_score["bleu"]]
                }
                temp_df = pd.DataFrame(data=df)
                temp_df.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, index=True)
