from experiments.utils import run_with_hyper_params_search

hyper_params_dist = dict(
    alg=['RND', 'POP', 'SPOP', 'SKNN', 'MARKOV'],
    dataset=[
        'RR-5',
        'RSC15_64',
    ],
    only_new=[False, True],
)

run_with_hyper_params_search(hyper_params_dist, gpu=False, nohup=True, max_processes=4)
