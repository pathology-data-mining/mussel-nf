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
p.add_argument('--workflow_id', required=False, default='reef')
p.add_argument('--output_prefix')
p.add_argument('results_dir')
args = p.parse_args()

output_csv = args.output_prefix + "_manifest.csv"
output_wide_csv = args.output_prefix + "_manifest_wide.csv"

workflow_id = args.workflow_id

dfs = []

results_dir = Path(args.results_dir)

features_pt = [{'slide_id': file.stem.split('.')[0],
                'workflow_id': workflow_id,
                'key': f"{file.parents[1].name}_features_tensor_path",
                'value': file.relative_to(results_dir)} for file in results_dir.glob("features/**/*.features.pt")]

tiles = [{'slide_id': file.stem.split('.')[0],
            'workflow_id': workflow_id,
            'key': "tiles_h5_path",
            'value': file.relative_to(results_dir)} for file in results_dir.glob("tiles/**/*.patch.h5")]

filter_tiles = [{'slide_id': file.stem.split('.')[0],
            'workflow_id': workflow_id,
            'key': "filtered_tiles_h5_path",
            'value': file.relative_to(results_dir)} for file in results_dir.glob("filter_tiles/**/*.patch.h5")]
df = pd.DataFrame.from_records(features_pt + tiles + filter_tiles)

df.to_csv(output_csv, index=False, header=False)
df.drop_duplicates(['slide_id', 'workflow_id', 'key']) \
    .pivot(index=['slide_id', 'workflow_id'], columns='key', values='value') \
    .to_csv(output_wide_csv)

