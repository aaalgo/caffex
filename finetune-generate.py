#!/usr/bin/env python
import sys 
import os
import logging
import subprocess
import simplejson as json
from jinja2 import Environment, FileSystemLoader

params_json = open("config.json", "r").read()
params = json.loads(params_json)

TMPL_ROOT = os.path.join(os.path.dirname(__file__), "templates")
TMPL_DIR = os.path.join(TMPL_ROOT, params['template'])

TO_BE_REPLACED = ["train_val.prototxt", "deploy.prototxt", "solver.prototxt"]

cmd = "cp -r %s/* ./" % TMPL_DIR
subprocess.check_call(cmd, shell=True)

templateLoader = FileSystemLoader(searchpath="./" )
templateEnv = Environment(loader=templateLoader)

for path in TO_BE_REPLACED:
    template = templateEnv.get_template(path)
    out = template.render(params)
    with open(path, 'w') as f:
        f.write(out)

