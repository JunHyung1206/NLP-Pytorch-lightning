from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
import Instances.instance as instance
import wandb

import Utils.utils as utils


def train(args, conf):

    project_name = conf.wandb.project
    dataloader, model = instance.new_instance(conf)
    wandb_logger = WandbLogger(project=project_name)

    save_path = f"{conf.path.save_path}{conf.model.model_name}_{wandb_logger.experiment.name}/"

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_dict[conf.utils.early_stop_monitor]["monitor"],
                mode=utils.monitor_dict[conf.utils.early_stop_monitor]["mode"],
                patience=conf.utils.patience,
            ),
            utils.best_save(
                save_path=save_path,
                top_k=conf.utils.top_k,
                monitor=utils.monitor_dict[conf.utils.best_save_monitor]["monitor"],
                mode=utils.monitor_dict[conf.utils.best_save_monitor]["mode"],
                filename="{epoch}-{val_pearson}",  # best 모델 저장시에 filename 설정
            ),
        ],
    )
    trainer.fit(model=model, datamodule=dataloader)
    test_pearson = trainer.test(model=model, datamodule=dataloader)
    wandb.finish()

    test_pearson = test_pearson[0]["test_pearson"]
    trainer.save_checkpoint(f"{save_path}epoch={conf.train.max_epoch-1}-test_pearson={test_pearson}.ckpt")


def k_fold_train(args, conf):
    project_name = conf.wandb.project
    results = []
    num_folds = conf.k_fold.num_folds
    for k in range(num_folds):
        k_dataloader, k_model = instance.kfold_new_instance(conf, k)
        name_ = f"{k+1}th_fold"
        wandb_logger = WandbLogger(project=project_name, name=name_)
        save_path = f"{conf.path.save_path}{conf.model.model_name}/{args.config}_K_fold/"  # 모델 저장 디렉터리명에 wandb run name 추가
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=conf.train.max_epoch,
            log_every_n_steps=1,
            logger=wandb_logger,
        )
        trainer.fit(model=k_model, datamodule=k_dataloader)
        test_pearson = trainer.test(model=k_model, datamodule=k_dataloader)
        wandb.finish()
        test_pearson = test_pearson[0]["test_pearson"]

        results.append(test_pearson)
        trainer.save_checkpoint(f"{save_path}{k+1}-Fold.ckpt")

    score = sum(results) / num_folds
    print(score)


def continue_train(args, conf):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_dict[conf.utils.early_stop_monitor]["monitor"],
                mode=utils.monitor_dict[conf.utils.early_stop_monitor]["mode"],
                patience=conf.utils.patience,
            ),
            utils.best_save(
                save_path=save_path,
                top_k=conf.utils.top_k,
                monitor=utils.monitor_dict[conf.utils.best_save_monitor]["monitor"],
                mode=utils.monitor_dict[conf.utils.best_save_monitor]["mode"],
                filename="{epoch}-{val_pearson}",  # best 모델 저장시에 filename 설정
            ),
        ],
    )
    dataloader, model, args, conf = instance.load_instance(args, conf)
    wandb_logger = WandbLogger(project=conf.wandb.project)

    save_path = f"{conf.path.save_path}{conf.model.model_name}_{wandb_logger.experiment.name}/"

    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1, logger=wandb_logger)
    trainer.fit(model=model, datamodule=dataloader)
    test_pearson = trainer.test(model=model, datamodule=dataloader)
    wandb.finish()

    test_pearson = test_pearson[0]["test_pearson"]
    trainer.save_checkpoint(f"{save_path}epoch={conf.train.max_epoch-1}-test_pearson={test_pearson}.ckpt")
