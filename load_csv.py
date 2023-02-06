import pandas as pd
from torch.utils.data import Dataset

from augmentation import add_ex, random_ex_swap


class UMRF_Prompts(Dataset):

    def __init__(self, input):
        self.input = input

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx: int):
        input = self.input[idx]
        return input


class UMRF(Dataset):

    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx: int):
        input = self.input[idx]
        target = self.target[idx]
        return input, target


# Training Dataset
train_umrf_frame = pd.read_csv('umrf.csv')

prompts = train_umrf_frame['prompt'].values

train_umrf_dataset = UMRF_Prompts(prompts)

# Validation Dataset
valid_umrf_frame = pd.read_csv('valid_umrf.csv')

prompts = valid_umrf_frame['prompt'].values
targets = valid_umrf_frame['target'].values

valid_umrf_dataset = UMRF(prompts, targets)

test = add_ex(train_umrf_dataset[0], train_umrf_dataset, m=0.0)

# print(test)

# print('\n')
# print('\n')

test2 = add_ex(test, train_umrf_dataset, m=0.0)

# print(test2)

# print('\n')
# print('\n')

test3 = add_ex(test2, train_umrf_dataset, m=0.0)

# print(test3)

test4 = random_ex_swap(test3, m=0.8)

# print(test4)