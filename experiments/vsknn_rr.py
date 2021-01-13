from experiments.utils import run_with_hyper_params_search

hyper_params_dist = dict(
    alg=['VSKNN'],
    dataset=[
        'RR-5',
    ],
    k=[1500],
    sample_size=[2500],
    #    similarity=['cosine', 'jaccard', 'sorensen_dice'],
    #    sampling=['random', 'recent', 'common'],
    similarity=['cosine'],
    sampling=['recent'],
    weighting=['same'],
    weighting_score=['linear'],
    idf_weighting=[10],
    normalize=['True'],
    only_new=[False, True]
)

run_with_hyper_params_search(hyper_params_dist, gpu=False, nohup=False, max_processes=4)
