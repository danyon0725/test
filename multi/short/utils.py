import logging
import random

import numpy as np
import torch

from multi.short.processor import (
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features
)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# tensor를 list 형으로 변환하기위한 함수
def to_list(tensor):
    return tensor.detach().cpu().tolist()

# dataset을 load 하는 함수
def load_examples(args, tokenizer, evaluate=False, output_examples=False):
    '''

    :param args: 하이퍼 파라미터
    :param tokenizer: tokenization에 사용되는 tokenizer
    :param evaluate: 평가나 open test시, True
    :param output_examples: 평가나 open test 시, True / True 일 경우, examples와 features를 같이 return
    :return:
    examples : max_length 상관 없이, 원문으로 각 데이터를 저장한 리스트
    features : max_length에 따라 분할 및 tokenize된 원문 리스트
    dataset : max_length에 따라 분할 및 학습에 직접적으로 사용되는 tensor 형태로 변환된 입력 ids
    '''

    # processor 선언
    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

    # open test 시
    if evaluate:
        examples = processor.get_dev_examples(args.dev_file_path)
    # 학습 시
    else:
        examples = processor.get_train_examples(args.train_file_path)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
    )

    if output_examples:
        return dataset, examples, features
    return dataset
