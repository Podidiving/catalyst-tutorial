from argparse import ArgumentParser

import os
import pandas as pd
import safitty
from sklearn.model_selection import train_test_split

from catalyst.utils import (
    map_dataframe
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--df',
        required=True,
        type=str,
        help='path to data-frame'
    )
    parser.add_argument(
        '-l', '--labeling',
        required=True,
        type=str,
    )
    parser.add_argument(
        '-o', '--out-path',
        required=True,
        type=str
    )
    parser.add_argument(
        '-s', '--seed',
        required=False,
        type=int,
        default=42
    )
    parser.add_argument(
        '-t', '--test',
        required=False,
        type=float,
        default=0.2
    )

    return parser.parse_args()


def prepare_splits(args):
    tag2class = dict(safitty.load(args.labeling))
    df_with_labels = map_dataframe(
        pd.read_csv(args.df),
        tag_column="class",
        class_column="label",
        tag2class=tag2class,
        verbose=False
    )
    train_data, val_data = train_test_split(
        df_with_labels,
        random_state=args.seed,
        test_size=args.test
    )
    train_data.to_csv(
        os.path.join(args.out_path, 'train.csv'),
        index=False
    )
    val_data.to_csv(
        os.path.join(args.out_path, 'valid.csv'),
        index=False
    )


if __name__ == '__main__':
    args = parse_args()
    prepare_splits(args)
