#!/usr/bin/env bash

python train.py \
  -M ../../../.mindnlp/model/facebook/hubert-large-ll60k \
  -P ../../../.mindnlp/model/facebook/hubert-large-ls960-ft \
  --schema linear \
  --lr 2e-4 \
  --device CPU \
  --device_id 1 \
  --debug
