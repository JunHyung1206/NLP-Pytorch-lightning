from Instances.Dataloaders.dataset import Dataset
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, StratifiedShuffleSplit

import pandas as pd
import transformers
import torch
import pytorch_lightning as pl
import Utils.utils as utils

# train, dev, test, predict
class Dataloader_Ver1(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.model_name = conf.model.model_name
        self.batch_size = conf.train.batch_size
        self.shuffle = conf.data.shuffle

        self.train_path = conf.path.train_path
        self.dev_path = conf.path.dev_path
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
            text = self.tokenizer.sep_token.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)

            sep_tokens_idx = [idx for idx, value in enumerate(outputs["input_ids"]) if value == self.tokenizer.sep_token_id]  # sep 토큰의 위치
            outputs["token_type_ids"] = [0] * len(outputs["input_ids"])  # [0,0, ... ,0]으로 초기화
            for i in range(sep_tokens_idx[0], sep_tokens_idx[1] + 1):
                outputs["token_type_ids"][i] = 1

            data.append(outputs)

        if swap:
            for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
                text = self.tokenizer.sep_token.join([item[text_column] for text_column in self.text_columns[::-1]])
                outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)

                sep_tokens_idx = [idx for idx, value in enumerate(outputs["input_ids"]) if value == self.tokenizer.sep_token_id]  # sep 토큰의 위치
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])  # [0,0, ... ,0]으로 초기화
                for i in range(sep_tokens_idx[0], sep_tokens_idx[1] + 1):
                    outputs["token_type_ids"][i] = 1
                data.append(outputs)

        return data

    def preprocessing(self, data, swap):
        data = data.drop(columns=self.delete_columns)
        try:
            targets = data[self.target_columns].values.tolist()
            if swap:
                targets += data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data, swap)
        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":

            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data, self.swap)
            val_inputs, val_targets = self.preprocessing(val_data, self.swap)

            print("train data len : ", len(train_inputs))
            print("valid data len : ", len(val_inputs))

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)

        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

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


# (train+dev), test, predict
class Dataloader_Ver2(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.model_name = conf.model.model_name
        self.batch_size = conf.train.batch_size
        self.shuffle = conf.data.shuffle
        self.train_ratio = conf.data.train_ratio
        self.seed = conf.utils.seed

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
            text = self.tokenizer.sep_token.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)

            sep_tokens_idx = [idx for idx, value in enumerate(outputs["input_ids"]) if value == self.tokenizer.sep_token_id]
            outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            for i in range(sep_tokens_idx[0], sep_tokens_idx[1] + 1):
                outputs["token_type_ids"][i] = 1

            data.append(outputs)

        if swap:
            for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
                text = self.tokenizer.sep_token.join([item[text_column] for text_column in self.text_columns[::-1]])
                outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)

                sep_tokens_idx = [idx for idx, value in enumerate(outputs["input_ids"]) if value == self.tokenizer.sep_token_id]
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
                for i in range(sep_tokens_idx[0], sep_tokens_idx[1] + 1):
                    outputs["token_type_ids"][i] = 1
                data.append(outputs)

        return data

    def preprocessing(self, data, swap):
        data = data.drop(columns=self.delete_columns)
        try:
            targets = data[self.target_columns].values.tolist()
            if swap:
                targets += data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data, swap)
        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)
            split = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=self.seed)
            for train_idx, val_idx in split.split(total_data, total_data["binary-label"]):
                train_data = total_data.loc[train_idx]
                val_data = total_data.loc[val_idx]

            train_inputs, train_targets = self.preprocessing(train_data, self.swap)
            val_inputs, val_targets = self.preprocessing(val_data, self.swap)

            print("train data len : ", len(train_inputs))
            print("valid data len : ", len(val_inputs))

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)

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
