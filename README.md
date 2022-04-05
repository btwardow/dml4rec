# DML4Rec

Code for the paper:  
_**Metric Learning for Session-based Recommendations**_  
 *Bartłomiej Twardowski, Paweł Zawistowski, Szymon Zaborowski* \
 European Conference on Information Retrieval 2021 \
 [arxiv](https://arxiv.org/abs/2101.02655), [ECIR2021](https://link.springer.com/chapter/10.1007/978-3-030-72113-8_43) 

```bibtex
@inproceedings{twardowski2021metric,
  title={Metric Learning for Session-Based Recommendations},
  author={Twardowski, Bart{\l}omiej and Zawistowski, Pawe{\l} and Zaborowski, Szymon},
  booktitle={European Conference on Information Retrieval},
  pages={650--665},
  year={2021},
  organization={Springer}
}
```

## Environment set-up

Environment is based on Conda distribution. 
All dependencies are in `environment.yml` file.
For Docker version check out `docker` directory.

### Create env

To create new environment check out repo and type: 
```
conda env create --file environment.yml --name dml4rec
```

*Notice:* set appropriate version of your CUDA driver for `cudatoolkit` in `environment.yml`.

### Environment activation/deactivation 

```
conda activate dml4rec
conda deactivate
```

## Datasets preparation

See README in `data/` directory.

## Running single experiment

Available recommendation algorithms to run:

| alg           | Class                             |
| ------------- |:----------------------------------|
| RND           | RandomRecommender                 |
| POP           | MostPopularRecommender            |
| SPOP          | PopularityInSessionRecommender    |
| SKNN          | SessionKnnRecommender             |
| VSKNN         | VMSessionKnnRecommender           |
| MARKOV        | MarkovRecommender                 |
| DML           | DMLSessionRecommender             |


All hyper-parmas are in the `init()` method of the recommender class (like in sklearn). Simples file will have `alg` key with selected algorithm and optional parameters. 

Sample experiments files are in `examples/` directory. 

To run single experiment type: 
```bash
./run.sh experiments/experiment.py examples/dml-RSC15_64-MaxPool-SDMLAllLoss.json
```

To select GPU device use it with `CUDA_VISIBLE_DEVICES=X` prefix.

Full results will be stored in json file in `results/` directory and only some evaluation metrics will be presented in logs as below: 

```
2021-01-06 16:54:45,811 - rec.eval - Test prediction time: 25.048 sec.
2021-01-06 16:54:45,811 - rec.eval - Eveluation - test sessions num: 31352
2021-01-06 16:54:45,815 - rec.eval - Evaluation - ground truth items num: 93510
2021-01-06 16:54:45,817 - rec.eval - Evaluation - predictions items num: 627040
2021-01-06 16:54:45,873 - rec.eval - Unique items predicted: 8605
2021-01-06 16:54:46,023 - rec.eval - REC@20: 0.519
2021-01-06 16:54:52,709 - rec.eval - HR@20: 0.6478
2021-01-06 16:54:53,789 - rec.eval - Evaluation time: 7.915 sec.
```

## Running baselines

All simple baselines can be run at once using script:

```
./run.sh experiments/baseline.py
```

## Unittests

To run all unit tests type: 

```
pytest tests/
```
