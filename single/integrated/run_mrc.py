import argparse
import logging
import os

import torch
from attrdict import AttrDict
from single.integrated.model import ElectraForQuestionAnswering
from single.integrated.utils import init_logger, set_seed
from transformers import ElectraTokenizer, ElectraConfig

from single.integrated.main_functions import train, evaluate, only_scoring

from nsml import DATASET_PATH
from torch import nn
def create_model(args):
    # 모델 초기 가중치를 init_weight로 지정하거나 파라미터로 입력한 checkpoint로 지정
    config = ElectraConfig.from_pretrained(
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
    )
    # tokenizer는 pre-trained된 것을 불러온다기보다 저장된 모델에 추가 vocab 등을 불러오는 과정
    tokenizer = ElectraTokenizer.from_pretrained(
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
        do_lower_case=args.do_lower_case,
    )
    model = ElectraForQuestionAnswering.from_pretrained(
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
        config=config,
        from_tf= False
    )
    model.to(args.device)
    return model, tokenizer

def main(cli_args):
    # 파라미터 업데이트
    args = AttrDict(vars(cli_args))
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    logger = logging.getLogger(__name__)

    # logger 및 seed 지정
    init_logger()
    set_seed(args)

    # 모델 불러오기
    model, tokenizer = create_model(args)

    if args.do_train:
        train(args, model, tokenizer, logger)
    if args.do_eval:
        evaluate(args, model, tokenizer)
    if args.scoring:
        only_scoring(args, tokenizer)
if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()


    file_flag = 'baseline'
    # Directory
    cli_parser.add_argument("--model_name_or_path", type=str, default="google/electra-base-discriminator")
    cli_parser.add_argument("--output_dir", type=str, default="./model")
    # cli_parser.add_argument("--train_file_path", type=str, default="../../data/refine_test.json")
    # cli_parser.add_argument("--dev_file_path", type=str, default="../../data/refine_sample.json")
    cli_parser.add_argument("--train_file_path", type=str, default=os.path.join(DATASET_PATH, 'train',"sent_test.json"))
    cli_parser.add_argument("--dev_file_path", type=str, default=os.path.join(DATASET_PATH, 'train',"sent_test.json"))

    # Model Hyper Parameter
    cli_parser.add_argument("--max_seq_length", type=int, default=512)
    cli_parser.add_argument("--doc_stride", type=int, default=128)
    cli_parser.add_argument("--max_query_length", type=int, default=64)
    cli_parser.add_argument("--max_answer_length", type=int, default=200)
    cli_parser.add_argument("--n_best_size", type=int, default=10)

    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=5e-5)
    cli_parser.add_argument("--train_batch_size", type=int, default=32)
    cli_parser.add_argument("--eval_batch_size", type=int, default=16)
    cli_parser.add_argument("--num_train_epochs", type=int, default=5)

    cli_parser.add_argument("--threads", type=int, default=12)
    cli_parser.add_argument("--save_steps", type=int, default=200)
    cli_parser.add_argument("--logging_steps", type=int, default=200)
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--max_sent", type=int, default=200)
    cli_parser.add_argument("--result_file", type=str, default="./result_file")
    cli_parser.add_argument("--weight_decay", type=float, default=0.0)
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    cli_parser.add_argument("--warmup_steps", type=int, default=0)
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--do_lower_case", type=bool, default=True)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    # For SQuAD v2.0 (Yes/No Question)
    cli_parser.add_argument("--version_2_with_negative", type=bool, default=False)
    cli_parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)

    # Running Mode
    cli_parser.add_argument("--from_init_weight", type=bool, default=True)
    cli_parser.add_argument("--do_train", type=bool, default=True)
    cli_parser.add_argument("--do_eval", type=bool, default=False)
    cli_parser.add_argument("--scoring", type=bool, default=False)

    cli_args = cli_parser.parse_args()
    print(cli_args)
    main(cli_args)
