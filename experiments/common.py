import os

from rec.dataset.dataset import Dataset
from rec.dataset.split import LastNPercentOfSessionsInDataset, RandomSessionSplitter
from rec.eval import PrecisionRecallAtN, PrecisionRecallUpToN, \
    MeanReciprocalRankUpToN, HitRateAtN, MeanReciprocalRankAtN, MeanAveragePrecisionAtN

n = 20

POSITIVE_EVENT_TYPES = {
    'SI': ['CLICK', 'VIEW'],  # SI dataset
    'RR': ['view', 'addtocart', 'transaction'],  # retailrocket_dataset
    'RS': ['VIEW']
}

synthetic_dataset_name = 'TEST10K'

DATASET_FILE_WITH_CONTEXT = 'dataset.pkl.gz'
DATASET_FILE_NO_CONTEXT = 'dataset_no_context.pkl.gz'


def datasets(dataset_names, load_only_clicked_items=False, dataset_root='data/dataset', no_context=True):
    for dataset_name in dataset_names:
        if dataset_name.startswith(synthetic_dataset_name):
            yield Dataset.generate_test_data(
                sessions_num=10000,
                items_num=1000,
                events_in_session=10,
                sessions_by_user=2,
                name=synthetic_dataset_name
            )
        else:
            dataset_filename = DATASET_FILE_NO_CONTEXT if no_context else DATASET_FILE_WITH_CONTEXT
            dataset_file_path = f'data/dataset/{dataset_name}/{dataset_filename}'
            if os.path.exists(dataset_file_path):
                yield Dataset.read_from_file(dataset_file_path)
            else:
                sessions_data_file = f'{dataset_root}/{dataset_name}/sessions.jsonl.gz'
                items_data_file = f'{dataset_root}/{dataset_name}/items.jsonl.gz'
                if no_context:
                    yield Dataset(dataset_name).load_with_no_context(sessions_data_file)

                yield Dataset(dataset_name).load_from_file(
                    sessions_data_file, items_data_file, load_only_clicked_items=load_only_clicked_items
                )


evaluation_measures = [
    PrecisionRecallAtN(n),
    MeanReciprocalRankAtN(n, single_next_item_only=True),
    PrecisionRecallUpToN(n),
    MeanReciprocalRankUpToN(n, single_next_item_only=True),
    HitRateAtN(n, single_next_item_only=True),
    MeanAveragePrecisionAtN(n)
]

# splitter = RandomSessionSplitter(train_ratio=0.98)
# splitter = TimestampSessionSplitter(3 * 24 * 60 * 60 * 1000)
# splitter = TimestampSessionSplitter(3 * 24 * 60 * 60)
splitter = LastNPercentOfSessionsInDataset(split_percent=0.1)
valid_splitter = RandomSessionSplitter(train_ratio=0.95)
