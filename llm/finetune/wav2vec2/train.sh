#!/usr/bin/env bash

python train.py \
  -M ../../../.mindnlp/model/facebook/wav2vec2-base \
  -P ../../../.mindnlp/model/facebook/wav2vec2-base-960h \
  --schema linear \
  --lr 2e-4 \
  --device CPU \
  --device_id 1 \
  --debug
