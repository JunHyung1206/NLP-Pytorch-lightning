import argparse

import random
import numpy as np
import torch

from omegaconf import OmegaConf

from Step import train, inference, sweep


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


# TODO
# K폴드는 최대한 단일 모델로 적정 하이퍼 파라미터를 얻은 뒤 사용
# 데이터 -> 모델 -> 하이퍼 파라미터 순으로 고려
# 백본 모델을 선정할 때, 그리고 모델 구조를 변경할 땐 최대한 근거를 가지고 가설을 설정한 뒤 실험하기
# 모델 구조를 변경할 때는 대조군을 꼭 갖기
# 하이퍼 파라미터 조정은 우선 grid로 대략적인 범위를 파악한 뒤 그다음 그 범위 내에서 베이지안 기법 적용
# 시퀀스를 자를 때 유동적으로도 자를 수 있으므로 고려 (버케팅)
# 스케줄러 람다 함수 작성할 공간 마련해두기 (util 밑에다 둘까 아니면 모델 밑에다 둘까 고민된다)
# 개인적으로 레이 한번 사용해보고 싶음

if __name__ == "__main__":

    args, conf = init()

    if args.mode == "train" or args.mode == "t":
        if conf.k_fold.use_k_fold:  # num_folds 변수 확인
            train.k_fold_train(args, conf)
        else:
            train.train(args, conf)

    elif args.mode == "continue" or args.mode == "c":
        if args.saved_model is None:
            print("경로를 입력해주세요")
        elif conf.k_fold.use_k_fold:
            print("K-Fold 추가 학습 불가능")
        else:
            train.continue_train(args, conf)

    elif args.mode == "exp" or args.mode == "e":
        exp_count = int(input("실험할 횟수를 입력해주세요 "))
        sweep.sweep(args, conf, exp_count)

    elif args.mode == "inference" or args.mode == "i":
        if args.saved_model is None:
            print("경로를 입력해주세요")
        else:
            if conf.k_fold.use_k_fold:  # num_folds 변수 확인
                inference.k_fold_inference(args, conf)
            else:
                inference.inference(args, conf)

    else:
        print("모드를 다시 설정해주세요 ")
        print("train        : t,\ttrain")
        print("continue     : c,\tcontinue")
        print("sweep        : e,\texp")
        print("inference    : i,\tinference")
