import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import Instances.instance as instance


def inference(args, conf):

    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)

    dataloader, model, args, conf = instance.load_instance(args, conf)

    test_pearson = trainer.test(model=model, datamodule=dataloader)
    test_pearson = test_pearson[0]["test_pearson"]

    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )

    predictions = list(float(i) for i in torch.cat(predictions))
    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv(f"output-{round(float(test_pearson), 4)}.csv", index=False)


def k_fold_inference(args, conf):
    test_list = []
    predictions_list = []
    for k in range(conf.k_fold.num_folds):
        trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)

        k_dataloader, k_model = instance.kfold_load_instance(args, conf, k)

        test_pearson = trainer.test(model=k_model, datamodule=k_dataloader)
        test_pearson = test_pearson[0]["test_pearson"]
        test_list.append(test_pearson)

        predictions = trainer.predict(
            model=k_model,
            datamodule=k_dataloader,
        )

        predictions = list(float(i) for i in torch.cat(predictions))
        predictions_list.append(predictions)

    total_predictions = np.stack(predictions_list, axis=0)
    average_predictions = total_predictions.sum(axis=0) / conf.k_fold.num_folds
    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = average_predictions

    score = sum(test_list) / conf.k_fold.num_folds
    output.to_csv(f"K_fold_output-{round(float(score), 4)}.csv", index=False)
