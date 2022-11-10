from Instances.Dataloaders.dataloaders import Dataloader_Ver1, Dataloader_Ver2
from Instances.Dataloaders.k_fold_dataloader import KFoldDataloader
from Instances.Models.model import Model, CustomModel_DenseNet

import streamlit as st
import transformers
import pytorch_lightning as pl
import torch
from Instances.Dataloaders.dataloaders import Dataloader_Streamlit


def new_instance(conf):
    dataloader = Dataloader_Ver1(conf)
    model = CustomModel_DenseNet(conf, dataloader.new_vocab_size())
    return dataloader, model


def load_instance(args, conf):
    dataloader, model = new_instance(conf)
    save_path = "/".join(args.saved_model.split("/")[:-1])

    # huggingface에 저장된 모델명을 parsing함
    # ex) 'klue/roberta-small'
    model_name = "/".join(args.saved_model.split("/")[1:-1]).split("_")[0]

    if args.saved_model.split(".")[-1] != "ckpt":
        exit("saved_model 파일 오류")

    model = model.load_from_checkpoint(args.saved_model)

    conf.path.save_path = save_path + "/"
    conf.model.model_name = "/".join(model_name.split("/")[1:])
    return dataloader, model, args, conf


def kfold_new_instance(conf, k):
    # def __init__(self, conf, k):
    k_dataloader = KFoldDataloader(conf, k)
    k_model = Model(conf, k_dataloader.new_vocab_size())
    return k_dataloader, k_model


def kfold_load_instance(args, conf, k):
    k_dataloader, k_model = kfold_new_instance(conf, k)

    # print(f"{args.saved_model}")
    model_name = "/".join(args.saved_model.split("/")[1:3])
    conf.model.model_name = model_name

    if args.saved_model.split(".")[-1] == "ckpt":
        exit("saved_model 파일 오류, k_fold 설정 확인!")
    k_model = k_model.load_from_checkpoint(args.saved_model + f"/{k+1}-Fold.ckpt")

    return k_dataloader, k_model


@st.cache(allow_output_mutation=True)
def load_model(args, conf):
    tokenizer = transformers.AutoTokenizer.from_pretrained(conf.model.model_name)
    _, model, _, _ = load_instance(args, conf)
    return model, tokenizer


def run_sts(sentence1, sentence2, conf, model):
    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)
    dataloader = Dataloader_Streamlit(conf, sentence1=sentence1, sentence2=sentence2)
    predictions = trainer.predict(model=model, datamodule=dataloader)
    print(predictions)
    result = predictions[0].item()
    if result > 5.0:
        result = 5.0
    elif result < 0.0:
        result = 0.0
    return result
