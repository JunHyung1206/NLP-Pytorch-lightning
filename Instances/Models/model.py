from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR


import transformers
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import Utils.utils as utils


class Model(pl.LightningModule):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = conf.model.model_name
        self.lr = conf.train.lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=self.model_name, num_labels=1)
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정
        self.loss_func = utils.loss_dict[conf.train.loss]
        self.use_freeze = conf.train.use_freeze

        if self.use_freeze:
            self.freeze()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.plm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)["logits"]
        return x

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids, labels)
        loss = self.loss_func(logits, labels.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids, labels)
        loss = self.loss_func(logits, labels.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), labels.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids, labels)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), labels.squeeze()))

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids, labels)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def freeze(self):
        for name, param in self.plm.named_parameters():
            freeze_list = []  # 얼릴 파라미터 이름을 리스트 안에 지정
            if name in freeze_list:
                param.requires_grad = False


class CustomModel_DenseNet(pl.LightningModule):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = conf.model.model_name
        self.lr = conf.train.lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=self.model_name)
        self.input_dim = transformers.AutoConfig.from_pretrained(self.model_name).d_model  # input vector
        self.hidden_dim = 1024

        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정
        self.loss_func = utils.loss_dict[conf.train.loss]
        self.use_freeze = conf.train.use_freeze

        self.Head = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.Head2 = nn.Sequential(nn.Linear(self.input_dim + self.hidden_dim, 1))

        if self.use_freeze:
            self.freeze()

    def forward(self, x):
        x = self.plm(x)[0]
        x = x[:, 0, :]  # cls 토큰만 추출
        y = self.Head(x)  # y: 1024
        x = torch.cat((x, y), dim=1)
        x = self.Head2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def freeze(self):
        for name, param in self.plm.named_parameters():
            freeze_list = []  # 얼릴 파라미터 이름을 리스트 안에 지정
            if name in freeze_list:
                param.requires_grad = False
