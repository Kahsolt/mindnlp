'''
Finetune HuBERT train script
'''
import os
from argparse import ArgumentParser
from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray
from datasets import load_dataset as load_dataset_hf, Split, Dataset as HFDataset

import mindspore
from mindspore import Tensor
from mindspore import context
from mindspore.dataset import GeneratorDataset, BatchDataset
from mindspore import ops, context
from mindspore.nn.optim import Adam
from mindspore._c_dataengine import CBatchInfo

from mindnlp.dataset import load_dataset
from mindnlp.engine import Trainer, TrainingArguments
from mindnlp._legacy.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp._legacy.metrics import Accuracy
from mindnlp.transformers.feature_extraction_utils import BatchFeature
from mindnlp.transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer


class ProcessedDataset:

    def __init__(self, dataset:HFDataset, processor:Wav2Vec2Processor):
        self.dataset = dataset
        self.processor = processor

        tokenizer: Wav2Vec2CTCTokenizer = self.processor.tokenizer
        self.pad_id = tokenizer(tokenizer.pad_token)['input_ids'][0]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx:Union[int, np.int64]) -> Tuple[ndarray]:
        if isinstance(idx, np.int64): idx = idx.item()

        sample = self.dataset[idx]
        processed: BatchFeature = self.processor(
            audio=sample['audio']['array'], 
            sampling_rate=sample['audio']['sampling_rate'], 
            text=sample['text'], 
            return_tensors='np',
        )
        return processed.input_values, processed.labels

    def collator_fn(self, inputs:List[ndarray], labels:List[ndarray], batch_info:CBatchInfo) -> List[ndarray]:
        inputs = [np.squeeze(e, axis=0) for e in inputs]
        labels = [np.squeeze(e, axis=0) for e in labels]

        pad_to = 'fixed'
        if pad_to == 'batch':
            maxlen_A = max([len(e) for e in inputs])
            maxlen_T = max([len(e) for e in labels])
        elif pad_to == 'dataset':
            maxlen_A = 470400
            maxlen_T = 363
        elif pad_to == 'fixed':
            maxlen_A = 16000 * 10
            maxlen_T = 100

        def process_A(e:ndarray, maxlen:int) -> ndarray:
            d = maxlen - e.shape[-1]
            if d == 0: return e
            if d > 0: return np.pad(e, (0, d), constant_values=e.mean())
            if d < 0: return e[:maxlen]
        def process_T(e:ndarray, maxlen:int) -> ndarray:
            d = maxlen - e.shape[-1]
            if d == 0: return e
            if d > 0: return np.pad(e, (0, d), constant_values=self.pad_id)
            if d < 0: return e[:maxlen]

        inputs = [process_A(e, maxlen_A) for e in inputs]
        labels = [process_T(e, maxlen_T) for e in labels]
        return inputs, labels


def make_batch_dataset(data:HFDataset, processor:Wav2Vec2Processor, batch_size:int, column_names:List[str]=['input_values', 'labels'], n_workers:int=None) -> BatchDataset:
    data_proc = ProcessedDataset(data, processor)
    data_gen = GeneratorDataset(data_proc, column_names=column_names)
    dataset = data_gen.batch(batch_size, per_batch_map=data_proc.collator_fn, num_parallel_workers=n_workers)
    print(f'>> [batch_size={batch_size}] n_samples: {len(data_gen)} => n_batches: {len(dataset)}')
    return dataset


def show_param_cnt(model:Wav2Vec2ForCTC):
    params = model.trainable_params()
    print('>> trainable param_cnt:', sum([p.numel() for p in params]))
    print('>> trainable params:', [name for name, p in model.parameters_and_names() if p.requires_grad])
    return params


def run_debug(args):
    model: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(args.model)
    processor = Wav2Vec2Processor.from_pretrained(args.processor, do_lower_case=True)

    local_path = '/home/daiyuxin/Kahsolt/librispeech_asr_dummy'
    path = 'hf-internal-testing/librispeech_asr_dummy'
    if os.path.exists(local_path):
        path = local_path
    data: HFDataset = load_dataset_hf(path, 'clean', split=Split.VALIDATION, trust_remote_code=True)
    dataset = make_batch_dataset(data, processor, args.batch_size, n_workers=1)

    if not 'stats max length':
        data_proc = ProcessedDataset(data, processor)
        data_gen = GeneratorDataset(data_proc, column_names=['input_values', 'labels'])
        maxlen_A, maxlen_T = 0, 0
        for A, T in data_gen:
            maxlen_A = max(maxlen_A, A.shape[-1])
            maxlen_T = max(maxlen_T, T.shape[-1])
        print('maxlen_A:', maxlen_A)
        print('maxlen_T:', maxlen_T)

    def test_infer():
        model.set_train(False)
        X, Y = next(iter(dataset))
        print('input_values.shape:', X.shape)
        print('labels.shape:', Y.shape)
        print('truth_ids:', Y)
        for it in Y:
            text = processor.decode(it)
            print(text)
        logits = model(X, labels=Y).logits
        print('logits.shape:', logits.shape)
        predicted_ids = ops.argmax(logits, dim=-1)
        print('predicted_ids:', predicted_ids)
        for it in predicted_ids:
            text = processor.decode(it)
            print(text)

    print('[Before train]')
    test_infer()
    test_infer()

    exit()

    if args.schema == 'linear':
        model.freeze_base_model()
    model.set_train(True)
    show_param_cnt(model)

    if not 'use trainer':       # seems buggy
        training_args = TrainingArguments(
            args.save_dir,
            overwrite_output_dir=True,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            logging_steps=10,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
    else:
        def forward_fn(X:Tensor, Y:Tensor) -> Tuple[Tensor, Tensor]:
            nonlocal model
            ret = model(X, labels=Y)
            return ret.loss, ret.logits

        params = [p for name, p in model.parameters_and_names() if p.requires_grad]
        optim = Adam(params, learning_rate=args.lr)
        grad_fn = mindspore.value_and_grad(forward_fn, None, optim.parameters, has_aux=True)

        steps = 0
        for epoch in range(args.epochs):
            for X, Y in dataset:
                (loss, logits), grads = grad_fn(X, Y)
                optim(grads)

                steps += 1
                if steps % 10 == 0:
                    print(f'>> steps {steps}')

    print('[After train]')
    test_infer()

    print('Done!')


def run(args):
    # Load model
    model: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(args.model)
    #model.config.layerdrop = -1    # 会导致动态图炸显存
    processor = Wav2Vec2Processor.from_pretrained(args.processor, do_lower_case=True)

    # Load dataset
    if args.dataset == 'librispeech_asr':
        train_data = load_dataset_hf(args.dataset, args.name, split=args.split, trust_remote_code=True)
        train_dataset = make_batch_dataset(train_data, processor, args.batch_size)
        val_data = load_dataset_hf(args.dataset, args.name, split=Split.VALIDATION, trust_remote_code=True)
        val_dataset = make_batch_dataset(val_data, processor, args.batch_size)
    else:
        raise NotImplementedError('>> add your own data source here')

    # Set validation metric and checkpoint callbacks
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_callback = CheckpointCallback(save_path=args.save_dir, ckpt_name='hubert', epochs=1, keep_checkpoint_max=2)
    best_model_callback = BestModelCallback(save_path=args.save_dir, ckpt_name='hubert', auto_load=True)

    # Initiate the trainable params
    if args.schema == 'linear':
        model.freeze_base_model()
    model.set_train(True)
    show_param_cnt(model)

    # Initiate the trainer
    training_args = TrainingArguments(
        args.save_dir,
        overwrite_output_dir=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[ckpt_callback, best_model_callback],
    )

    # Start training
    trainer.train()


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-M', '--model', default='facebook/wav2vec2-base', help='model hub from hugging face')
    parser.add_argument('-P', '--processor', default='facebook/wav2vec2-base-960h', help='processor hub from hugging face')
    parser.add_argument('-D', '--dataset', default='librispeech_asr', help='dataset hub from hugging face')
    parser.add_argument('--name', default='clean', help='dataset name')
    parser.add_argument('--split', default='train.100', help='dataset split for training')
    parser.add_argument('-S', '--schema', default='linear', choices=['linear', 'full'], help='finetune scheme for parameters')
    parser.add_argument('-B', '--batch_size', type=int, default=4)
    parser.add_argument('-E', '--epochs', type=int, default=2)
    parser.add_argument('-lr', '--lr', type=eval, default=2e-5)
    parser.add_argument('--save_dir', default='./checkpoint/hubert')
    parser.add_argument('--device', default='GPU', choices=['CPU', 'GPU', 'Ascend'])
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # sanity check
    if args.dataset == 'librispeech_asr':
        assert args.split in ['train.100', 'train.360']

    # global init
    context.set_context(device_target=args.device, device_id=args.device_id)
    if args.debug: context.set_context(pynative_synchronize=True)
    np.random.seed(args.seed)
    mindspore.set_seed(args.seed)

    return args


if __name__ == '__main__':
    args = get_args()

    if args.debug:
        run_debug(args)
    else:
        run(args)
