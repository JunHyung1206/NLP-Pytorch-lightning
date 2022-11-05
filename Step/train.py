from tqdm.auto import tqdm


import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import Instances.instance as instance


def train(args, conf):
    
    dataloader, model = instance.new_instance(conf)
    
    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, 'model.pt')
