'''
Finetune HuBERT eval script
'''
import os
from argparse import ArgumentParser
from functools import partial

from datasets import load_dataset, Split, Dataset

from mindspore.dataset import GeneratorDataset
from mindspore import nn, ops, context

from mindnlp.engine import Evaluator
from mindnlp.metrics import Accuracy
from mindnlp.transformers import (
    HubertPreTrainedModel,
    HubertForCTC,
    Wav2Vec2Processor,
)


def run(args):
    # Load model
    model: HubertForCTC = HubertForCTC.from_pretrained(args.model)
    processor = Wav2Vec2Processor.from_pretrained(args.processor, do_lower_case=True)

    # Load dataset
    if args.dataset == 'librispeech_asr':
        dataset_test = load_dataset(args.dataset, 'clean', split=Split.TEST, trust_remote_code=True)
    else:
        raise NotImplementedError('add your own data source here')

    # Set validation metric and checkpoint callbacks
    metric = Accuracy()

    # Initiate the evaluator
    evaluator = Evaluator(
        network=model,
        eval_dataset=dataset_test,
        metrics=metric)

    # Start evaluating
    evaluator.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-M', '--model', default='facebook/hubert-large-ll60k', help='base hub from hugging face')
    parser.add_argument('-P', '--processor', default='facebook/hubert-large-ls960-ft', help='base hub from hugging face')
    parser.add_argument('-D', '--dataset', default='librispeech_asr', help='dataset from hugging face')
    parser.add_argument('-S', '--split', default='train.100', help='training data split')
    parser.add_argument('--save_dir', default='./checkpoint/hubert')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    context.set_context(device_target=args.device, device_id=args.device_id)

    run(args)
