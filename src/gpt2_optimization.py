import os

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from umrf_dataset import UMRF
from prompt_score_maxing import JiangPrompt


def check_for_gpu() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def create_prompt(nl_instruction:str, image_data: str, umrf_exs: list) -> str:
    raise NotImplementedError


if __name__ == '__main__':
    device = check_for_gpu()

    umrf_data_path = os.getcwd() + '/umrf_data/*'
    umrf_dataset = UMRF(umrf_data_path)

    prompt_method = JiangPrompt(umrf_dataset)
    print(prompt_method.input_information)

    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # prompt = "hello my name is Selma"
    # prompt_tokenized = gpt2_tokenizer(prompt, return_tensors="pt").to(device)
    # outputs = gpt2_model.generate(**prompt_tokenized, return_dict_in_generate=True,
    #  output_scores=True, max_new_tokens=100)
    
    # # output_scores provide logits over GPT2 vocabulary size
    # # to make into logprobs, apply torch.softmax()
    # output_scores = outputs['scores']

    # # greedy decoding of most likely next token prediction
    # output_tokens = gpt2_tokenizer.decode(outputs['sequences'][0])




