import torch
from torch.utils.data import DataLoader
import transformers
from transformers import GPT2Tokenizer, GPT2Model

from umrf_dataset import UMRF


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

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2Model.from_pretrained("gpt2").to(device)
