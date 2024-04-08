# Exmaples of fine-tuning HuBERT

Full parameter fine-tuning or linear-probing the HuBERT model on the `librispeech_asr` dataset, resuming from pretrained checkpoint `facebook/hubert-large-ll60k` on dataset `Libri-Light`.

⚪ train

```shell
# default configs
python llm/finetune/hubert/train.py
# this is equal to
python llm/finetune/hubert/train.py \
  --model_hub facebook/hubert-large-ll60k \
  --batch_size 4 \
  --epochs 2
```

⚪ eval

```shell
# default configs
python llm/finetune/hubert/eval.py
# this is equal to
python llm/finetune/hubert/train.py \
  --model_hub facebook/hubert-large-ll60k \
```
