import pandas as pd
import torch
import pytorch_lightning as pl
import Instances.instance as instance


def inference(args, conf):

    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)

    dataloader, model, args, conf = instance.load_instance(args, conf)

    trainer.test(model=model, datamodule=dataloader)
    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )

    predictions = list(float(i) for i in torch.cat(predictions))  # 리스트화
    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv("output.csv", index=False)
