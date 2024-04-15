import evaluate
import pandas as pd

bleu_fn = evaluate.load("bleu")

BASE_PATH = "/vast/home/slwanna/git_repos/gpt_umrf_parser/2024_adv_robot_csv/2024_adv_robot_csv/"

file_names = ["codellama_minimal_id_test_trial_0.csv", "codellama_minimal_id_test_trial_1.csv", "codellama_minimal_id_test_trial_2.csv", "llama-7b_minimal_id_test_trial_0.csv", "llama-7b_minimal_id_test_trial_1.csv", "llama-7b_minimal_id_test_trial_2.csv", "mistral_minimal_id_test_trial_0.csv", "mistral_minimal_id_test_trial_1.csv", "mistral_minimal_id_test_trial_2.csv"]

for fn in file_names:
    df = pd.read_csv(BASE_PATH+fn)
    df["clean_bleu"] = df.apply(lambda x: bleu_fn.compute(predictions=[str(x.cleaned_pred)], references=[[x.graph]])['bleu'], axis=1)
    df.to_csv(BASE_PATH+"cleaned_bleu_"+fn)
