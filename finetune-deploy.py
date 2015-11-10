#!/usr/bin/env python
import sys 
import os
import logging
import subprocess
import simplejson as json
from jinja2 import Environment, FileSystemLoader

params_json = open("config.json", "r").read()
params = json.loads(params_json)

print """
1. create a directory
2. copy deploy.prototxt to caffe.model
   copy snapshot to caffe.params
   copy mean.binaryproto to caffe.mean
   create a blob file listing the blobs to extract
"""
