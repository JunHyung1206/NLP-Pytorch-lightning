import pandas as pd
from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = (
            self.inputs[idx]["input_ids"],
            self.inputs[idx]["attention_mask"],
            self.inputs[idx]["token_type_ids"],
        )
        if len(self.targets) == 0:
            return (
                torch.tensor(input_ids),
                torch.tensor(attention_mask),
                torch.tensor(token_type_ids),
            )
        else:
            return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)
