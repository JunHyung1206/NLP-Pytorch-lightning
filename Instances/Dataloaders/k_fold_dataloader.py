from Instances.Dataloaders.dataset import Dataset
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

import pandas as pd
import transformers
import torch
import pytorch_lightning as pl
import Utils.utils as utils


class KFoldDataloader(pl.LightningDataModule):
    def __init__(self, conf, k):
        super().__init__()
        self.model_name = conf.model.model_name
        self.batch_size = conf.train.batch_size
        self.shuffle = conf.data.shuffle
        self.k = k
        self.num_split = conf.k_fold.num_split
        self.seed = conf.utils.seed  # 랜덤 시드

        self.train_path = conf.path.train_path
        self.test_path = conf.path.test_path
        self.predict_path = conf.path.predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        if self.model_name in utils.tokenizer_dict["bert"]:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        elif self.model_name in utils.tokenizer_dict["electra"]:
            self.tokenizer = transformers.ElectraTokenizer.from_pretrained(self.model_name)
        elif self.model_name in utils.tokenizer_dict["roberta"]:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
        elif self.model_name in utils.tokenizer_dict["funnel"]:
            self.tokenizer = transformers.FunnelTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        self.tokenizer.model_max_length = 128

        tokens = ["<PERSON>"]
        self.new_token_count = self.tokenizer.add_tokens(tokens)
        self.swap = conf.data.swap

        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataframe, swap):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
            text = [item[text_column] for text_column in self.text_columns]  # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            outputs = self.tokenizer(*text, add_special_tokens=True, padding="max_length", truncation=True)
            data.append(outputs)

        if swap:  # swap 적용시 양방향 될 수 있도록
            for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
                text = [item[text_column] for text_column in self.text_columns[::-1]]
                outputs = self.tokenizer(*text, add_special_tokens=True, padding="max_length", truncation=True)
                data.append(outputs)

        return data

    def preprocessing(self, data, swap):
        data = data.drop(columns=self.delete_columns)  # 안쓰는 컬럼을 삭제합니다.
        try:
            targets = data[self.target_columns].values.tolist()  # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
            if swap:
                targets += data[self.target_columns].values.tolist()  # 한번 더 사용
        except:
            targets = []
        inputs = self.tokenizing(data, swap)  # 텍스트 데이터를 전처리합니다.
        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)
            kfold = KFold(n_splits=self.num_split, shuffle=self.shuffle, random_state=self.seed)
            all_splits = [d_i for d_i in kfold.split(total_data)]

            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            print("Number of splits: \n", self.num_split)
            print("Before Swap Train data len: \n", len(train_indexes))
            print("Before Swap Valid data len: \n", len(val_indexes))

            train_inputs, train_targets = self.preprocessing(total_data.loc[train_indexes], self.swap)
            valid_inputs, valid_targets = self.preprocessing(total_data.loc[val_indexes], self.swap)

            train_dataset = Dataset(train_inputs, train_targets)
            valid_dataset = Dataset(valid_inputs, valid_targets)

            print("After Swap Train data len: \n", len(train_inputs))
            print("After Swap Valid data len: \n", len(valid_inputs))

            self.train_dataset = train_dataset
            self.val_dataset = valid_dataset
        else:
            test_data = pd.read_csv(self.test_path)  # 평가데이터 준비
            predict_data = pd.read_csv(self.predict_path)  # 예측할 데이터 준비

            test_inputs, test_targets = self.preprocessing(test_data, False)
            predict_inputs, predict_targets = self.preprocessing(predict_data, False)

            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

    def new_vocab_size(self):
        return self.new_token_count + self.tokenizer.vocab_size
