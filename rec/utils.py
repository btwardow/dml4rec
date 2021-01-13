import time
import datetime


class Nop:
    def __call__(self, *args, **kwargs):
        return None


class NopWriter(object):
    def __getattr__(self, attr):
        return Nop()

    def __setattr__(self, attr, val):
        pass


tb = NopWriter()


def init_tb(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    global tb
    tb = SummaryWriter(log_dir)
    return tb


def current_milli_time():
    """
    Returns current time in milliseconds.
    :return: current milliseconds
    """
    return int(round(time.time() * 1000))


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def iso_timestamp_to_epoch(ts):
    """Takes ISO 8601 format(string) and converts into epoch time."""
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if 'T' in ts else "%Y-%m-%d %H:%M:%S.%f"
    if 'Z' in ts:
        ts = ts[:-1]
    dt = datetime.datetime.strptime(ts, fmt)
    dt = dt.replace(tzinfo=None)
    seconds = time.mktime(dt.timetuple()) + dt.microsecond // 1000000
    return int(seconds)


def seed_everything(seed=0, pytorch=False):
    import random
    import numpy as np
    import os

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if pytorch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
