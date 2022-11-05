from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
import Instances.instance as instance
import wandb

import Utils.utils as utils


def sweep(args, conf, exp_count):  # 메인에서 받아온 args와 실험을 반복할 횟수를 받아옵니다
    project_name = conf.wandb.project

    sweep_config = {
        "method": "bayes",  # random: 임의의 값의 parameter 세트를 선택, #bayes : 베이지안 최적화
        "parameters": {
            "lr": {
                # parameter를 설정하는 기준을 선택합니다. uniform은 연속적으로 균등한 값들을 선택합니다.
                "distribution": "uniform",
                "min": 1e-5,  # 최소값을 설정합니다.
                "max": 3e-5,  # 최대값을 설정합니다.
            },
        },
        "early_terminate": {  # 위의 링크에 있던 예시
            "type": "hyperband",
            "max_iter": 30,  # 프로그램에 대해 최대 반복 횟수 지정, min과 max는 같이 사용 불가능한듯
            "s": 2,
        },
        "metric": {"name": "test_pearson", "goal": "maximize"},  # pearson 점수가 최대화가 되는 방향으로 학습을 진행합니다.
    }

    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config
        conf.train.lr = config.lr
        dataloader, model = instance.new_instance(conf)

        wandb_logger = WandbLogger(project=project_name)
        save_path = f"{conf.path.save_path}{conf.model.model_name}_{wandb.run.name}/"

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

    sweep_id = wandb.sweep(
        sweep=sweep_config,  # config 딕셔너리를 추가합니다.
        project=project_name,  # project의 이름을 추가합니다.
    )

    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=exp_count)  # 실험할 횟수 지정
