#!/usr/bin/env python

from argparse import ArgumentParser
import os
import glob
from pathlib import Path
import yaml

import pandas as pd

p = ArgumentParser()
p.add_argument('--attr_yaml', type=Path)
p.add_argument('--output_csv')
p.add_argument('results_dir', nargs='+')
args = p.parse_args()

dfs = []

for results_dir in args.results_dir:
    results_dir = Path(results_dir)

    features_pt = [{'slide_id': file.stem.split('.')[0],
    'type': f"{file.parents[1].name}_features_tensor_urlpath",
    'value': file.resolve()} for file in results_dir.glob("**/*.features.pt")]

    tiles = [{'slide_id': file.stem.split('.')[0],
    'type': "tiles_h5_urlpath",
    'value': file.resolve()} for file in results_dir.glob("**/*.patch.h5")]

    df = pd.DataFrame.from_records(features_pt + tiles)

    if args.attr_yaml:
        with open(args.attr_yaml, 'r') as f:
            attrs = yaml.safe_load(f)
            attr_dfs = []
            for key, value in attrs.items():
                attr_df =  pd.DataFrame({"slide_id":df.slide_id.unique(), 'type': key, 'value': value})
                attr_dfs.append(attr_df)
            attr_df = pd.concat(attr_dfs)
            df = pd.concat([df, attr_df])

    dfs.append(df)

if len(dfs) > 0:
    pd.concat(dfs).to_csv(args.output_csv, index=False, header=False)
