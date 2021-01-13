from experiments.utils import run_with_hyper_params_search

# params are taken from https://rn5l.github.io/session-rec/index.html

hyper_params_dist = dict(
    alg=['VSKNN'],
    dataset=[
        'RSC15_64',
    ],
    k=[1000],
    sample_size=[5000],
    similarity=['cosine'],
    # sampling=['random', 'recent', 'common'],
    sampling=['recent'],
    weighting=['log'],
    weighting_score=['quadratic'],
    idf_weighting=[2],
    only_new=[False, True]
)

run_with_hyper_params_search(hyper_params_dist, gpu=False, nohup=False, max_processes=4)
