from Instances.Dataloaders.dataloaders import Dataloader_Ver1, Dataloader_Ver2
from Instances.Models.model import Model


def new_instance(conf):
    dataloader = Dataloader_Ver2(conf)
    model = Model(conf, dataloader.new_vocab_size())
    return dataloader, model


def kfold_new_instance(conf):
    pass


def load_instance(args, conf):
    dataloader, model = new_instance(conf)
    save_path = "/".join(args.saved_model.split("/")[:-1])

    # huggingface에 저장된 모델명을 parsing함
    # ex) 'klue/roberta-small'
    model_name = "/".join(args.saved_model.split("/")[1:-1]).split("_")[0]
    model = model.load_from_checkpoint(args.saved_model)

    conf.path.save_path = save_path + "/"
    conf.model.model_name = "/".join(model_name.split("/")[1:])
    return dataloader, model, args, conf
