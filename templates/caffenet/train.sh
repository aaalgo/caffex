#!/usr/bin/env sh

export GLOG_log_dir=log
export GLOG_logtostderr=1

SNAP=$1
if [ -z "$SNAP" ]
then
    nice caffe train --solver solver.prototxt --weights init.caffemodel $*
else
    shift
    nice caffe train -solver solver.prototxt -snapshot $SNAP $*
fi

