from Instances.Dataloaders.dataloaders import Dataloader_Ver1, Dataloader_Ver2
from Instances.Dataloaders.k_fold_dataloader import KFoldDataloader
from Instances.Models.model import Model, CustomModel_DenseNet


def new_instance(conf):
    dataloader = Dataloader_Ver2(conf)
    model = Model(conf, dataloader.new_vocab_size())
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
