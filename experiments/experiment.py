import datetime
import logging
import os
import pprint
import sys
import json

from rec.dataset.testcase_generator import SubsequentEventTestCaseGenerator
from rec.experiment import Experiment
from rec.recommender import *
from experiments.common import datasets, splitter, evaluation_measures, n, valid_splitter, POSITIVE_EVENT_TYPES
from rec.recommender.vsknn import VMSessionKnnRecommender
from rec.utils import seed_everything

print("Running params: {}".format(",".join(sys.argv)))

EXPERIMENT_ID = sys.argv[1] if len(sys.argv) > 1 else '{}_{}'.format(
    os.getpid(),
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)

# logging configuration
LOG_FILE_NAME = 'logs/{}.log'.format(EXPERIMENT_ID)

if len(sys.argv) > 2:
    LOG_FILE_NAME = sys.argv[2]

print('Logging to file: {}'.format(LOG_FILE_NAME))
LOGGING_FORMAT = '%(asctime)s %(name)-8s %(levelname)-1s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, filename=LOG_FILE_NAME)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

print("Params: ", ', '.join(sys.argv))
params_file = sys.argv[3]
print('Reading parameters from file: ', params_file)

hyper_params = json.load(open(params_file, 'r'))
pprint.pprint(hyper_params)

# only new
ONLY_NEW = False
if 'only_new' in hyper_params:
    ONLY_NEW = hyper_params['only_new']
    del hyper_params['only_new']
print('ONLY_NEW:', ONLY_NEW)

# results destination
RESULTS_DIR = hyper_params.get('results_dir', 'results')
print('RESULTS_DIR:', RESULTS_DIR)
if 'results_dir' in hyper_params:
    del hyper_params['results_dir']

# load dataset
dataset_name = hyper_params['dataset']
dataset_prefix = dataset_name[:2].upper()
del (hyper_params['dataset'])
dataset_no_context = 'encoder' not in hyper_params  # by default don't load context info.

dataset = next(datasets([dataset_name], no_context=dataset_no_context))

# filter sessions to MAX_SESSION_LEN
MAX_SESSION_LEN = 8 if dataset_prefix == 'SI' else 15
print(f'Dataset before filtering to {MAX_SESSION_LEN}:', dataset)
dataset.sessions = {
    u_id: {s_id: s
           for s_id, s in sessions.items() if len(s) <= MAX_SESSION_LEN}
    for u_id, sessions in dataset.sessions.items() if any(len(s) <= MAX_SESSION_LEN for s in sessions.values())
}

dataset._create_indexes()
print(f'Dataset after filtering to {MAX_SESSION_LEN}:', dataset)

positive_event_types = POSITIVE_EVENT_TYPES[dataset_prefix]

# selecting recommender
alg = hyper_params['alg']
del hyper_params['alg']

ALGOS = dict(
    RND=RandomRecommender,
    POP=MostPopularRecommender,
    SPOP=PopularityInSessionRecommender,
    SKNN=SessionKnnRecommender,
    VSKNN=VMSessionKnnRecommender,
    MARKOV=MarkovRecommender,
    DML=DMLSessionRecommender
)
assert alg in ALGOS, f'Wrong algorithm: {alg}!'

cls = ALGOS[alg]
hp_names = set(cls._get_param_names()).intersection(hyper_params.keys())
hp = {p: hyper_params[p] for p in hp_names}
print(f'Hyper-params for {alg}:', hp)
unused_hp = hyper_params.keys() - hp.keys()
if len(unused_hp) > 0:
    print('WARNING: Unused hyper-params:', unused_hp)
recommender = cls(**hp)

# running the experiment
experiment = Experiment(
    dataset,
    recommender,
    train_test_splitter=splitter,
    test_case_generator=SubsequentEventTestCaseGenerator(
        only_new=ONLY_NEW,
        positive_event_types=positive_event_types,
    ),
    evaluation_measures=evaluation_measures,
    iteration_num=1,
    top_n=n,
    test_batch_size=10,
    train_valid_splitter=valid_splitter,
    experiment_id=EXPERIMENT_ID,
    use_tensorboard=True,
    results_dir=RESULTS_DIR
)

experiment.run_and_save_results()
