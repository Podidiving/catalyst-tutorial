# catalyst-tutorial


[catalyst](https://github.com/catalyst-team/catalyst)

[more generic classification pipeline](https://github.com/catalyst-team/classification)


## Data preparation


1. Download dataset
```bash
download-gdrive 1eAk36MEMjKPKL5j9VWLvNTVKk4ube9Ml artworks.tar.gz
extract-archive artworks.tar.gz &>/dev/null
```

2. Create DataFrame:
```bash
catalyst-data tag2label \
    --in-dir=./data/dataset \
    --out-dataset=./data/dataset.csv \
    --out-labeling=./data/labeling.json \
    --tag-column=class
```

3. Prepare train / val splits
```bash
python3 utils/prepare_splits.py \
    --df=./data/dataset.csv \
    --labeling=./data/labeling.json \
    --out-path=./data/
```


## Training


1. `CUDA_VISIBLE_DEVICES="<YOUR_DEVICES>" catalyst-dl run --configs training/configs/train_config.yml`


## While training


1. Tensorboard training logs are available at `<logdir>/train_log`
1. Tensorboard validation logs are available at `<logdir>/valid_log`


## After training

1. Weights for final model (which is the best, according to `<main metric>` on validation) are available at `<logdir>/checkpoints/best.pth`
