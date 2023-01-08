"""
This code is only to test the huggingface transformer
GPT-2 tokenizer/model prediction.

No prompt optimization occurs here.
"""


import os

import torch
from torch.utils.data import DataLoader, random_split
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from umrf_dataset import UMRF
from prompt import Prompt
from prompt_score_maxing import JiangPrompt


def check_for_gpu() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(230)
    else:
        device = 'cpu'
    return device


if __name__ == '__main__':
    torch.manual_seed(230)
    device = check_for_gpu()

    umrf_data_path = os.getcwd() + '/umrf_data/*'
    umrf_dataset = UMRF(umrf_data_path)

    training_exs, validation_exs = random_split(umrf_dataset, [20, 7],
                                                generator=torch.Generator().manual_seed(42))

    print('Step 2: Create Prompt Obj')
    prompts = Prompt(input_information=training_exs,
                     validation_exs=validation_exs)

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    prompt = "hello my name is selma"

    prompt_tokenized = gpt2_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = gpt2_model.generate(**prompt_tokenized, return_dict_in_generate=True,
                                  output_scores=True, max_new_tokens=100)

    # output_scores provide logits over GPT2 vocabulary size
    # to make into logprobs, apply torch.softmax()
    output_scores = outputs['scores']

    # greedy decoding of most likely next token prediction
    output_tokens = gpt2_tokenizer.decode(outputs['sequences'][0])

    print(output_tokens)
