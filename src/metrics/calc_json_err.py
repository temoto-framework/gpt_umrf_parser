import evaluate
import json
import pandas as pd

bleu_fn = evaluate.load("bleu")

CLEANED_BLEU_PREFIX = "cleaned_bleu_"
BASE_PATH = "/vast/home/slwanna/git_repos/gpt_umrf_parser/2024_adv_robot_csv/2024_adv_robot_csv/"

file_names = ["codellama_minimal_id_test_trial_0.csv", "codellama_minimal_id_test_trial_1.csv", "codellama_minimal_id_test_trial_2.csv", "llama-7b_minimal_id_test_trial_0.csv", "llama-7b_minimal_id_test_trial_1.csv", "llama-7b_minimal_id_test_trial_2.csv", "mistral_minimal_id_test_trial_0.csv", "mistral_minimal_id_test_trial_1.csv", "mistral_minimal_id_test_trial_2.csv"]

def check_json(x):
    try:
        json.loads(x.cleaned_pred)
        return 1
    except:
        return 0

for fn in file_names:
    df = pd.read_csv(BASE_PATH+CLEANED_BLEU_PREFIX+fn)
    df["json_err"] = df.apply(lambda x: check_json(x))
    df.to_csv(BASE_PATH+"cleaned_bleu_and_json_err_"+fn)
