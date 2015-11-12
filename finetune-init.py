#!/usr/bin/env python
import sys 
import os
import logging
import argparse
import simplejson as json

parser = argparse.ArgumentParser(description='init finetune directory.')
parser.add_argument('dir', nargs=1)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)

outd = args.dir[0]

os.mkdir(outd)

params = {
        "template": "caffenet",
        "backend": "PICPOC",
        "train_source": "train.poc",
        "train_batch": 32,
        "val_source": "val.poc",
        "val_batch": 10,
        "val_batches": 19,
        "fc_name": "fc8x",
        "num_output": 20,
        "lr_mult_filter": 10,
        "lr_mult_bias": 20,
        "val_interval": 100,
        "snapshot_interval": 200,
        "base_lr": 0.001,
        "stepsize": 20000,
        "max_iter": 100000,
        "device": "GPU",
	"data_type": "PicPocData"
}

params_json = json.dumps(params, sort_keys=False, indent=4 * ' ')
open(os.path.join(outd, 'config.json'), 'w').write(params_json)

