from tqdm.auto import tqdm

import transformers
import torch
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

    def forward(self, x):
        x = self.plm(x)["logits"]
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
