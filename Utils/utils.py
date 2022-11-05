import Instances.Models.loss as loss


loss_dict = {
    "l1": loss.L1_loss,
    "mse": loss.MSE_loss,
    "bce": loss.BCEWithLogitsLoss,
    "rmse": loss.RMSE_loss,
}
monitor_dict = {
    "val_loss": {"monitor": "val_loss", "mode": "min"},
    "val_pearson": {"monitor": "val_pearson", "mode": "max"},
}


tokenizer_dict = {
    "bert": [
        "klue/roberta-small",
        "klue/roberta-base",
        "klue/roberta-large",
    ],
    "electra": [
        "monologg/koelectra-base-v3-discriminator",
        "monologg/koelectra-base-finetuned-sentiment",
    ],
    "roberta": ["xlm-roberta-base"],
    "funnel": ["kykim/funnel-kor-base"],
}
