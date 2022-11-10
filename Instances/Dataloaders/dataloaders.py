from Instances.Dataloaders.dataset import Dataset
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, StratifiedShuffleSplit

import pandas as pd
import transformers
import torch
import pytorch_lightning as pl
import Utils.utils as utils

# train, dev, test, predict 따로 있는 버전
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
            text = self.tokenizer.special_tokens_map["sep_token"].join(
                [item[text_column] for text_column in self.text_columns]
            )  # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
            data.append(outputs["input_ids"])

        if swap:  # swap 적용시 양방향 될 수 있도록
            for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
                text = self.tokenizer.special_tokens_map["sep_token"].join([item[text_column] for text_column in self.text_columns[::-1]])
                outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
                data.append(outputs["input_ids"])

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

            train_data = pd.read_csv(self.train_path)  # 학습 데이터와 검증 데이터셋을 호출합니다
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data, self.swap)  # 학습데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data, self.swap)  # 검증데이터 준비

            print("train data len : ", len(train_inputs))
            print("valid data len : ", len(val_inputs))

            self.train_dataset = Dataset(train_inputs, train_targets)  # train 데이터만 shuffle을 적용해줍니다
            self.val_dataset = Dataset(val_inputs, val_targets)  # 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다4

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


# train, test, predict
class Dataloader_Ver2(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.model_name = conf.model.model_name
        self.batch_size = conf.train.batch_size
        self.shuffle = conf.data.shuffle
        self.train_ratio = conf.data.train_ratio  # train, dev 셋 나눌 비율
        self.seed = conf.utils.seed  # 랜덤 시드

        self.train_path = conf.path.train_path
        self.test_path = conf.path.test_path
        self.predict_path = conf.path.predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        ## TODO : 이름에 맞는 토크나이저 찾기로 수정
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
            text = self.tokenizer.special_tokens_map["sep_token"].join(
                [item[text_column] for text_column in self.text_columns]
            )  # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
            data.append(outputs)  # 바꿔준 부분

        if swap:  # swap 적용시 양방향 될 수 있도록
            for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
                text = self.tokenizer.special_tokens_map["sep_token"].join([item[text_column] for text_column in self.text_columns[::-1]])
                outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
                data.append(outputs)  # 바꿔준 부분

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
            split = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=self.seed)
            for train_idx, val_idx in split.split(total_data, total_data["binary-label"]):
                train_data = total_data.loc[train_idx]
                val_data = total_data.loc[val_idx]

            train_inputs, train_targets = self.preprocessing(train_data, self.swap)  # 학습데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data, self.swap)  # 검증데이터 준비

            print("train data len : ", len(train_inputs))
            print("valid data len : ", len(val_inputs))

            self.train_dataset = Dataset(train_inputs, train_targets)  # train 데이터만 shuffle을 적용해줍니다
            self.val_dataset = Dataset(val_inputs, val_targets)  # 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
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


class Dataloader_Streamlit(pl.LightningDataModule):
    def __init__(self, conf, sentence1, sentence2):
        super().__init__()
        self.model_name = conf.model.model_name
        self.batch_size = conf.train.batch_size
        self.shuffle = conf.data.shuffle
        self.train_ratio = conf.data.train_ratio  # train, dev 셋 나눌 비율
        self.seed = conf.utils.seed  # 랜덤 시드

        self.train_path = conf.path.train_path
        self.test_path = conf.path.test_path
        self.predict_path = conf.path.predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.sentence1 = sentence1
        self.sentence2 = sentence2

        ## TODO : 이름에 맞는 토크나이저 찾기로 수정
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
            text = self.tokenizer.special_tokens_map["sep_token"].join(
                [item[text_column] for text_column in self.text_columns]
            )  # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
            data.append(outputs["input_ids"])

        if swap:  # swap 적용시 양방향 될 수 있도록
            for idx, item in tqdm(dataframe.iterrows(), desc="tokenizing", total=len(dataframe)):
                text = self.tokenizer.special_tokens_map["sep_token"].join([item[text_column] for text_column in self.text_columns[::-1]])
                outputs = self.tokenizer(text, add_special_tokens=True, padding="max_length", truncation=True)
                data.append(outputs["input_ids"])

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
            pass
            # total_data = pd.read_csv(self.train_path)
            # split = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=self.seed)
            # for train_idx, val_idx in split.split(total_data, total_data["binary-label"]):
            #     train_data = total_data.loc[train_idx]
            #     val_data = total_data.loc[val_idx]

            # train_inputs, train_targets = self.preprocessing(train_data, self.swap)  # 학습데이터 준비
            # val_inputs, val_targets = self.preprocessing(val_data, self.swap)  # 검증데이터 준비

            # print("train data len : ", len(train_inputs))
            # print("valid data len : ", len(val_inputs))

            # self.train_dataset = Dataset(train_inputs, train_targets)  # train 데이터만 shuffle을 적용해줍니다
            # self.val_dataset = Dataset(val_inputs, val_targets)  # 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
        else:
            # test_data = pd.read_csv(self.test_path)  # 평가데이터 준비
            data = [{"id": 0, "sentence_1": self.sentence1, "sentence_2": self.sentence2}]
            predict_data = pd.DataFrame(data)

            # test_inputs, test_targets = self.preprocessing(test_data, False)
            predict_inputs, predict_targets = self.preprocessing(predict_data, False)

            # self.test_dataset = Dataset(test_inputs, test_targets)
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
