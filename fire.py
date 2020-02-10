#!/usr/bin/env python
# coding: utf-8

# Author: Marc van Oudheusden
# https://github.com/vanoudh/fire


from subprocess import call
from collections import defaultdict
from sagemaker.session import Session
import sagemaker
import io
import json
import hashlib
import boto3
import time
import sys
import os
import argparse


FIRE_PATH = '.fire'


def split(line, sep=' '):
    return [t.strip() for t in line.split(sep) if len(t) > 0]


def process_line(line, info):
    if 'get_ipython' in line:
        return None
    if '#fire-file' in line:
        _, name, key = split(line)
        info['keys'].append(key.strip())
        return f"{name} = '/opt/ml/input/data/training/{key}'\n"
    if '#fire-insert' in line:
        index = line.find('#fire-insert')
        return line[len('#fire-insert'):].strip() + '\n'
    if '#fire-' in line:
        tokens = split(line)
        _, name = split(tokens[0], '-')
        info[name.lower()] = info[name.lower()] + tokens[1:]
        return None
    return line


def to_json(o, bucket, key):
    with io.StringIO() as f:
        json.dump(o, f)
        f.seek(0)
        boto3.resource('s3').Object(bucket, key).put(Body=f.getvalue())

        
def factory(container):
    tokens = split(container.lower(), '-')
    if len(tokens) == 1:
        tokens.append(None)
    name, version = tokens
    if name == 'pytorch':
        from sagemaker.pytorch import PyTorch
        return PyTorch, version
    if name == 'tensorflow':
        from sagemaker.tensorflow import TensorFlow
        return TensorFlow, version
    if name == 'mxnet':
        from sagemaker.mxnet import MXNet
        return MXNet
    if name == 'sklearn':        
        from sagemaker.sklearn import SKLearn
        return SKLearn, version
    raise NotImplementedError(name)

    
def make_manifest_input(bucket, key, manifest):
    to_json(manifest, bucket, key)
    s3_input = sagemaker.session.s3_input(
        s3_data=f's3://{bucket}/{key}', 
        s3_data_type='ManifestFile'
    )
    return s3_input


parser = argparse.ArgumentParser(description='SageMaker notebook runner')
parser.add_argument('notebook_path', type=str, nargs=1, help='notebook to run')
parser.add_argument('--dryrun', default=False, action='store_true', help='prepare the job but do not send it')
args = parser.parse_args()
#print(args)

notebook_path = args.notebook_path[0]
dryrun = args.dryrun

filename = notebook_path.split('.')[0]
script_path = os.path.join(FIRE_PATH, filename + '.py')
fire_script_path = os.path.join(FIRE_PATH, filename + '_fire.py')

call(f'mkdir -p {FIRE_PATH}'.split())
call(f'jupyter nbconvert {notebook_path} --to script --output {os.path.join(FIRE_PATH, filename)}'.split())

info = defaultdict(list)

print('\nThe following script will be sent to SageMaker for execution:\n')

with open(script_path, 'r') as f:
    with open(fire_script_path, 'w') as fout:
        line = f.readline()
        while line:
            try:
                line_out = process_line(line, info)
            except ValueError as e:
                print(line)
                raise e
            if line_out:
                fout.write(line_out)
                print(line_out, end='')
            line = f.readline()

print('\nJob parameters:\n')

for k, v in info.items():
    print(f'{k:<12} : {v}')

if 'requirements' in info:
    with open(os.path.join(FIRE_PATH, 'requirements.txt'), 'w') as f:
        f.write('\n'.join(info['requirements']))

        
modules = ' '.join(info['modules'])
call(f'cp {modules} {FIRE_PATH}'.split())
    
Constructor, version = factory(info['container'][0])

estimator = Constructor(entry_point=filename + '_fire.py',
                        source_dir=FIRE_PATH,
                        role=sagemaker.get_execution_role(),
                        train_instance_count=1, 
                        train_instance_type=info['instance'][0],
                        framework_version=version
                        )

bucket = info['bucket'][0]
manifest = [{'prefix': f's3://{bucket}/'}] + info['keys']

default_bucket = Session().default_bucket()
input_obj = make_manifest_input(default_bucket, f'fire-manifests/{time.time()}.json', manifest)

if dryrun:
    print('\nExiting (--dryrun)\n')
    exit(0)
    
estimator.fit({'training': input_obj}) 
