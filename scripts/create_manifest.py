#!/usr/bin/env python

from argparse import ArgumentParser
import os
import glob
from pathlib import Path
import yaml
import hashlib
import json

import pandas as pd

p = ArgumentParser()
p.add_argument('--attr_yaml', type=Path)
p.add_argument('--workflow_id', required=False)
p.add_argument('--output_prefix')
p.add_argument('results_dir', nargs='+')
args = p.parse_args()

output_csv = args.output_prefix + "_manifest.csv"
output_attr_csv = args.output_prefix + "_params.csv"

output_wide_csv = args.output_prefix + "_manifest_wide.csv"
output_attr_wide_csv = args.output_prefix + "_params_wide.csv"

if args.workflow_id:
    workflow_id = args.workflow_id
else:
    workflow_id = hashlib.md5(json.dumps(args.attr_yaml, sort_keys=True).encode('utf-8')).hexdigest()
    workflow_id = workflow_id[-8:]

dfs = []

for results_dir in args.results_dir:
    results_dir = Path(results_dir)

    features_pt = [{'slide_id': file.stem.split('.')[0],
                    'workflow_id': workflow_id,
                    'key': f"{file.parents[1].name}_features_tensor_path",
                    'value': file.resolve()} for file in results_dir.glob("**/*.features.pt")]

    tiles = [{'slide_id': file.stem.split('.')[0],
              'workflow_id': workflow_id,
              'key': "tiles_h5_path",
              'value': file.resolve()} for file in results_dir.glob("**/*.patch.h5")]

    filter_tiles = [{'slide_id': file.stem.split('.')[0],
              'workflow_id': workflow_id,
              'key': "filtered_features_h5_path",
              'value': file.resolve()} for file in results_dir.glob("**/*.filtered_features.h5")]

    df = pd.DataFrame.from_records(features_pt + tiles + filter_tiles)
    dfs.append(df)

if len(dfs) > 0:
    df = pd.concat(dfs)
    df.to_csv(output_csv, index=False, header=False)
    df.drop_duplicates(['slide_id', 'workflow_id', 'key']).pivot(index=['slide_id', 'workflow_id'], columns='key', values='value').to_csv(output_wide_csv)

with open(args.attr_yaml, 'r') as f:
    attrs = yaml.safe_load(f)
    attr_df = pd.DataFrame.from_dict(attrs, orient='index')
    attr_df = attr_df.reset_index()
    attr_df.columns = ['key', 'value']
    attr_df.insert(0, 'workflow_id', workflow_id)
    attr_df.to_csv(output_attr_csv, header=False, index=False)
    attr_df.pivot(index='workflow_id', columns='key', values='value').to_csv(output_attr_wide_csv)

