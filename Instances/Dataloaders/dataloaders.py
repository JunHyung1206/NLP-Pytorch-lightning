from Instances.Dataloaders.dataset import Dataset
from tqdm.auto import tqdm

import pandas as pd
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

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
        self.predict_path = conf.path.test_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.model_max_length = 128
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns]) # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):       
        data = data.drop(columns=self.delete_columns) # 안쓰는 컬럼을 삭제합니다.       
        try:
            targets = data[self.target_columns].values.tolist() # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        except:
            targets = []     
        inputs = self.tokenizing(data) # 텍스트 데이터를 전처리합니다.
        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            
            train_data = pd.read_csv(self.train_path) # 학습 데이터와 검증 데이터셋을 호출합니다
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data) # 학습데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data) # 검증데이터 준비

           
            self.train_dataset = Dataset(train_inputs, train_targets)  # train 데이터만 shuffle을 적용해줍니다
            self.val_dataset = Dataset(val_inputs, val_targets) # 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
        else:
            test_data = pd.read_csv(self.test_path) # 평가데이터 준비
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path) # 예측할 데이터 준비
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    
    

# train, dev, test, predict 따로 있는 버전
class Dataloader_Ver2(pl.LightningDataModule):  
    def __init__(self, conf):
        super().__init__()
        self.model_name = conf.model.model_name
        self.batch_size = conf.train.batch_size
        self.shuffle = conf.data.shuffle

        self.train_path = conf.path.train_path
        self.dev_path = conf.path.dev_path
        self.test_path = conf.path.test_path
        self.predict_path = conf.path.test_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.model_max_length = 128
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns]) # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):       
        data = data.drop(columns=self.delete_columns) # 안쓰는 컬럼을 삭제합니다.       
        try:
            targets = data[self.target_columns].values.tolist() # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        except:
            targets = []     
        inputs = self.tokenizing(data) # 텍스트 데이터를 전처리합니다.
        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            
            train_data = pd.read_csv(self.train_path) # 학습 데이터와 검증 데이터셋을 호출합니다
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data) # 학습데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data) # 검증데이터 준비

           
            self.train_dataset = Dataset(train_inputs, train_targets)  # train 데이터만 shuffle을 적용해줍니다
            self.val_dataset = Dataset(val_inputs, val_targets) # 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
        else:
            test_data = pd.read_csv(self.test_path) # 평가데이터 준비
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path) # 예측할 데이터 준비
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)