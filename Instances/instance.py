from Instances.Dataloaders.dataloaders import Dataloader_Ver1, Dataloader_Ver2
from Instances.Models.model import Model


def new_instance(conf):
    dataloader = Dataloader_Ver2(conf)
    model = Model(conf, dataloader.new_vocab_size())
    return dataloader, model


def kfold_new_instance(conf):
    pass
