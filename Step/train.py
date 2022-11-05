from tqdm.auto import tqdm
from pytorch_lightning.loggers import WandbLogger

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import Instances.instance as instance


def train(args, conf):

    project_name = conf.wandb.project
    dataloader, model = instance.new_instance(conf)
    wandb_logger = WandbLogger(project=project_name)

    save_path = f"{conf.path.save_path}{conf.model.model_name}_{wandb_logger.experiment.name}/"

    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1, logger=wandb_logger)
    trainer.fit(model=model, datamodule=dataloader)
    test_pearson = trainer.test(model=model, datamodule=dataloader)
    test_pearson = test_pearson[0]["test_pearson"]
    trainer.save_checkpoint(f"{save_path}Epoch_{conf.train.max_epoch}-TestPearson_{test_pearson}.ckpt")


def k_fold_train(args, conf):
    pass


def continue_train(args, conf):
    pass
