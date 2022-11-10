import pandas as pd
from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = (
            self.inputs["input_ids"][idx],
            self.inputs["attention_mask"][idx],
            self.inputs["token_type_ids"][idx],
        )
        if len(self.targets) == 0:  # 정답이 있다면 else문을, 없다면 if문을 수행합니다
            return (input_ids, attention_mask, token_type_ids)
        else:
            return input_ids, attention_mask, token_type_ids, torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)  # 입력하는 개수만큼 데이터를 사용합니다
