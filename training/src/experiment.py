import collections

import numpy as np
import safitty

import torch
import torch.nn as nn

from catalyst.contrib.data.cv import ImageReader
from catalyst.data import Augmentor
from catalyst.data import (
    BalanceClassSampler,
    ListDataset,
    ReaderCompose,
    ScalarReader,
)
from catalyst.dl import ConfigExperiment
from catalyst.utils import read_csv_data

from .augmentor import (
    pre_transforms,
    hard_transforms,
    post_transforms,
    compose
)


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage == "finetune":
            for param in model_.parameters():
                param.requires_grad = False
            for param in model_.fc.parameters():
                param.requires_grad = True
        elif stage == "fulltrain":
            for param in model_.parameters():
                param.requires_grad = True
        return model_

    @staticmethod
    def get_transforms(
            stage: str = None,
            dataset: str = None
    ):
        if dataset == 'train':
            train_transforms = compose([
                pre_transforms(),
                hard_transforms(),
                post_transforms()
            ])
            train_data_transforms = Augmentor(
                dict_key="image",
                augment_fn=lambda x: train_transforms(image=x)["image"]
            )
            return train_data_transforms
        elif dataset == 'valid':
            valid_transforms = compose([pre_transforms(), post_transforms()])
            valid_data_transforms = Augmentor(
                dict_key="image",
                augment_fn=lambda x: valid_transforms(image=x)["image"]
            )
            return valid_data_transforms
        else:
            raise NotImplementedError

    def get_datasets(
        self,
        stage: str,
        datapath: str = None,
        in_csv: str = None,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        in_csv_infer: str = None,
        train_folds: str = None,
        valid_folds: str = None,
        tag2class: str = None,
        class_column: str = None,
        tag_column: str = None,
        folds_seed: int = 42,
        n_folds: int = 5,
        one_hot_classes: int = None,
        balance_strategy: str = "upsampling",
    ):
        datasets = collections.OrderedDict()
        tag2class = safitty.load(tag2class) if tag2class is not None else None

        df, df_train, df_valid, df_infer = read_csv_data(
            in_csv=in_csv,
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            train_folds=train_folds,
            valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            seed=folds_seed,
            n_folds=n_folds,
        )

        open_fn = [
            ImageReader(
                input_key="filepath", output_key="image", rootpath=datapath
            ),
            ScalarReader(
                input_key="label",
                output_key="targets",
                default_value=-1,
                dtype=np.int64,
            ),
        ]

        if one_hot_classes:
            open_fn.append(
                ScalarReader(
                    input_key="label",
                    output_key="targets_one_hot",
                    default_value=-1,
                    dtype=np.int64,
                    one_hot_classes=one_hot_classes,
                )
            )

        open_fn = ReaderCompose(readers=open_fn)

        for source, mode in zip(
            (df_train, df_valid, df_infer), ("train", "valid", "infer")
        ):
            if source is not None and len(source) > 0:
                dataset = ListDataset(
                    source,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(
                        stage=stage, dataset=mode
                    ),
                )
                if mode == "train":
                    labels = [x["label"] for x in source]
                    sampler = BalanceClassSampler(
                        labels, mode=balance_strategy
                    )
                    dataset = {"dataset": dataset, "sampler": sampler}
                datasets[mode] = dataset

        return datasets
