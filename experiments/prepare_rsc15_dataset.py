import os

import numpy as np
import pandas as pd
import datetime as dt
import rec.model as model
import gzip

from rec.dataset.dataset import Dataset

PATH_TO_ORIGINAL_DATA = 'data/rsc15/raw/'
PATH_TO_PROCESSED_DATA = 'data/dataset/rsc15/'
sessions_jsonl = PATH_TO_PROCESSED_DATA + 'sessions.jsonl'
sessions_gz = sessions_jsonl + '.gz'

blocksize = 2 << 16


def filter_data(data):
    session_lengths = data.groupby(model.EVENT_SESSION_ID).size()
    data = data[np.in1d(data[model.EVENT_SESSION_ID], session_lengths[session_lengths>1].index)]
    item_supports = data.groupby(model.EVENT_ITEM).size()
    data = data[np.in1d(data[model.EVENT_ITEM], item_supports[item_supports>=5].index)]
    session_lengths = data.groupby(model.EVENT_SESSION_ID).size()
    return data[np.in1d(data[model.EVENT_SESSION_ID], session_lengths[session_lengths>=2].index)]


def lookup(s):
    dates = {date:dt.datetime.strptime(date,'%Y-%m-%dT%H:%M:%S.%fZ').timestamp() for date in s.unique()}
    return s.map(dates)


def save_gz(source, dest):
    # TODO: this method is have a bug
    with open(source, 'rb') as sp:
        with gzip.open(dest, 'wb+') as dp:
            while True:
                block = sp.read(blocksize)
                if block == '':
                    break
                dp.write(block)
    return


def save_dataset(dataset, name):
    dataset.to_json(sessions_jsonl, lines=True, orient='records')
    save_gz(sessions_jsonl, sessions_gz)
    os.remove(sessions_jsonl)
    ds = Dataset(name).load_with_no_context(sessions_gz)
    os.makedirs(f'{PATH_TO_PROCESSED_DATA}/{name}')
    ds.write_to_file(f'{PATH_TO_PROCESSED_DATA}/{name}/dataset_no_context.pkl.gz')
    os.remove(sessions_gz)
    os.remove(sessions_jsonl)


if __name__ == '__main__':

    print('reading data')
    data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'rsc15-clicks.dat', sep=',', header=None, usecols=[0, 1, 2],
                       dtype={0: np.int32, 1: str, 2: np.int64})
    data.columns = [model.EVENT_SESSION_ID, 'TimeStr', model.EVENT_ITEM]
    print('processing timestr')
    data[model.TIMESTAMP] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    del (data['TimeStr'])

    print('dataset stats')
    no_sessions = len(data[model.EVENT_SESSION_ID].unique())
    no_events = len(data)
    print(no_sessions, no_events)


    print('filtering data')
    data = filter_data(data)

    len_64 = len(data) // 64
    data.sort_values([model.EVENT_SESSION_ID, model.TIMESTAMP], inplace=True)

    session_data = list(data[model.EVENT_SESSION_ID].values)
    lenth = int(len(session_data) // 64)
    session_data = session_data[-lenth:]
    for i in range(len(session_data)):
        if session_data[i] != session_data[i + 1]:
            break

    data_64 = data.reset_index()
    data_64 = data_64[-lenth + i + 1:]

    print('saving datasets')
    save_dataset(data)
    save_dataset(data_64)
