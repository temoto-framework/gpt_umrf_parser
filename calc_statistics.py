import evaluate
import pandas as pd

bleu_fn = evaluate.load("bleu")

"""
Analysis across all three trials using minimal task instruction
"""
codellama_minimal_csv_paths = [
    "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_1/CodeLlama-7b-Instruct-hf_minimal_id_test_output_generations.csv",
    "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_2/CodeLlama-7b-Instruct-hf_minimal_id_test_output_generations.csv",
    "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_3/CodeLlama-7b-Instruct-hf_minimal_id_test_output_generations.csv",
]

codellama_df_1 = pd.read_csv(codellama_minimal_csv_paths[0])
codellama_df_1["bleu_score"] = codellama_df_1.apply(lambda x: bleu_fn.compute([x.prediction], [x.graph]), axis=1)
# bleu_score_1 = bleu_fn.compute(predictions=codellama_df_1["prediction"], references=[codellama_df_1["graph"]])

codellama_df_2 = pd.read_csv(codellama_minimal_csv_paths[1])
# bleu_score_2 = bleu_fn.compute(predictions=codellama_df_2["prediction"], references=[codellama_df_2["graph"]])

codellama_df_3 = pd.read_csv(codellama_minimal_csv_paths[2])
# bleu_score_3 = bleu_fn.compute(predictions=codellama_df_3["prediction"], references=[codellama_df_3["graph"]])

# TODO: find average? find std. dev?
# bleu_score_list = [bleu_score_1, bleu_score_2, bleu_score_3]

# TODO: calc num valid jsons?

# llama_minimal_csv_paths = [
#     "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_1/Llama-2-7b-hf_minimal_id_test_output_generations.csv",
#     "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_2/Llama-2-7b-hf_minimal_id_test_output_generations.csv",
#     "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_3/Llama-2-7b-hf_minimal_id_test_output_generations.csv",
# ]
#
# llama_minimal_csv_paths = [
#     "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_1/Mistral-7B-Instruct-v0.2_minimal_id_test_output_generations.csv",
#     "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_2/Mistral-7B-Instruct-v0.2_minimal_id_test_output_generations.csv",
#     "/home/selmawanna/PycharmProjects/robotics_adv_robo_2024/generated_outputs/experiment_1/trial_3/Mistral-7B-Instruct-v0.2_minimal_id_test_output_generations.csv",
# ]

