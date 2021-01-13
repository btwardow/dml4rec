Dataset Preparation
===================

## RETAILROCKET

Download dataset from https://www.kaggle.com/retailrocket/ecommerce-dataset to `dataset/`. Name file as `ecommerce-dataset.zip`. 

```
cd data
mkdir -p retailrocket/raw
cd retailrocket/raw
unzip ../../ecommerce-dataset.zip
sort -k1 -n -t, events.csv > sorted_events.csv
sort -k1 -n -t, item_properties_part1.csv > sorted_item_properties_part1.csv
sort -k1 -n -t, item_properties_part2.csv > sorted_item_properties_part2.csv
```

then use `data/prepare_retailrocket_dataset.py` script to prepare RR-{1..5} datasets.


## RSC15

Download dataset file `res15.zip` from [here](https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0) (this is part of work in this [repository](https://github.com/rn5l/session-rec)) to `data` directory and extract it.

Then use the script:
```
PYTHONPATH='.' python data/prepare_rsc15_dataset.py
```
