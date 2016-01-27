#!/usr/bin/env python
import sys 
import os
import re
import shutil
import logging
import argparse
import subprocess
import simplejson as json

parser = argparse.ArgumentParser(description='')
parser.add_argument('input', nargs=1)
parser.add_argument('output', nargs=1)
parser.add_argument('tmp', nargs=1)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)

input = args.input[0]
output = args.output[0]
tmp = args.tmp[0]

if False:
    if not os.path.isdir(input):
        logging.error("%s is not directory" % input)
        sys.exit(1)
    if os.path.exists(output):
        logging.error("%s already exists" % output)
        sys.exit(1)
    if os.path.exists(tmp):
        logging.error("%s already exists" % tmp)
        sys.exit(1)

    bin_dir = os.path.dirname(__file__)

    subprocess.check_call("%s %s" % (os.path.join(bin_dir, "finetune-init.py"), tmp), shell=True)
    subprocess.check_call("load-caffex %s %s -U" % (input, tmp), shell=True)
    subprocess.check_call("cd %s; %s" % (tmp, os.path.join(bin_dir, "finetune-generate.py")), shell=True)
    subprocess.check_call("cd %s; ./train.sh 2> train.log" % tmp, shell=True)
    pass

perf = []
best_snap = -1
best_acc = 0
with open(os.path.join(tmp, "train.log")) as f:
    # look for lines like below
    # I0127 11:24:15.227892 27000 solver.cpp:340] Iteration 552, Testing net (#0)
    # I0127 11:24:15.283869 27000 solver.cpp:408]     Test net output #0: accuracy = 0.975
    re1 = re.compile("Iteration (\d+), Testing")
    re2 = re.compile("accuracy = (.+)")
    while True:
        l = f.readline()
        if not l:
            break
        m = re1.search(l)
        print m
        if m:
            it = int(m.group(1))
            l = f.readline()
            m = re2.search(l)
            acc = float(m.group(1))
            perf.append((it, acc))
            if acc > best_acc:
                best_acc = acc
                best_snap = it
            pass
        pass
    pass
# now we have performance data
print perf
# find best one
print "BEST ACCURACY", best_acc, "AT SNAPSHOT", best_snap
        
os.mkdir(output)
shutil.copy(os.path.join(tmp, "deploy.prototxt"),
            os.path.join(output, "caffe.model"))
shutil.copy(os.path.join(tmp, "mean.binaryproto"),
            os.path.join(output, "caffe.mean"))
shutil.copy(os.path.join(tmp, snapshots, "caffenet_train_iter_%d.caffemodel" % best_snap),
            os.path.join(output, "caffe.params"))
with open(output, "blobs") as f:
    f.write("prob\n")


        
        



# generate lists
#
