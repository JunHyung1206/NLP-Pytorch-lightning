import argparse

import random
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf

from Step import train, inference

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", required=True)
    parser.add_argument("--config", "-c", type=str, default="base_config")
    
    parser.add_argument(
        "--saved_model",
        "-s",
        default=None,
        help="저장된 모델의 파일 경로를 입력해주세요. 예시: save_models/klue/roberta-small/epoch=?-step=?.ckpt 또는 save_models/model.pt",
    )
    args, _ = parser.parse_known_args()
    conf = OmegaConf.load(f"./config/{args.config}.yaml")

    SEED = conf.utils.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return args, conf


if __name__ == "__main__":
    
    args, conf = init()
    
    if args.mode == "train" or args.mode == "t":
        if conf.k_fold.use_k_fold: # num_folds 변수 확인
            print("K-Fold Train")
        else:
            train.train(args, conf)

    elif args.mode == "continue" or args.mode == "c":
        if args.saved_model is None:
            print("경로를 입력해주세요")
        else:
            print("Continue") # k-fold는 추가 학습 고려 X

    elif args.mode == "exp" or args.mode == "e":
        exp_count = int(input("실험할 횟수를 입력해주세요 "))
        print("Sweep")

    elif args.mode == "inference" or args.mode == "i":
        if args.saved_model is None:
            print("경로를 입력해주세요")
        else:
            print("Inference")
    else:
        print("모드를 다시 설정해주세요 ")
        print("train     : t,\ttrain")
        print("exp       : e,\texp")
        print("inference : i,\tinference")
