import torch

from experiments.utils import run_with_hyper_params_search

hyper_params_dist = dict(
    alg=['DMLSession'],
    epochs=[150],  # patience should kick-in and stop this
    common_items_emb=[True],
    emb_dim=[400],
    s_net=[
        {
            'name': 'TagSpace'
        },
        {
            'name': 'CNN1D'
        },
        {
            'name': 'MaxPool',
            'activ': 'tanh'
        },
        {
            'dropout1_rate': 0.0,
            'hidden_size': 600,
            'name': 'RNN'
        },
        {
            'name': 'TextCNN',
            'filter_sizes': '1,3,5',
            'num_filters': 700,
            'dropout_rate': 0.2
        },
    ],
    optimizer=[{
        'lr': 0.001,
        'name': 'Adam',
        'weight_decay': 0.0
    }],
    max_len=[15],
    sampler=[
        {
            'name': 'SessionPosNegSampler',
            'batch_size': 32,
            'pos_len': 8,
            'neg_len': 8
        }, {
            'name': 'TripletLoss',
            'normalize': True,
            's_margin': 0.3
        }
    ],
    max_emb_norm=[1.0],
    loss=[{
        "label_softmax": True,
        "name": "SDMLAllLoss",
        "smoothing_parameter": 0.2
    }],
    dataset=[
        'RR-5',
        'RSC15_64'
    ]
)

run_with_hyper_params_search(hyper_params_dist, gpu=torch.cuda.is_available(), nohup=False, max_processes=1)
