import json
import os
import time

import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid

from rec.utils import current_milli_time


def _run_cmd_and_get_stdout(cmd_list):
    import subprocess
    r = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
    return r[0].decode('UTF-8').split('\n')


def occupied_gpus():
    import re
    smiOut = _run_cmd_and_get_stdout(["nvidia-smi"])
    r = re.compile('[ ]*')
    return list(map(int, [r.split(l)[1] for l in smiOut if 'python' in l]))


def get_pid_for_experiment_parma_file(param_file_name):
    ps_output = _run_cmd_and_get_stdout(['ps', '-elf'])
    sh_script_ps_out = [l for l in ps_output if param_file_name in l and 'python' in l]
    if len(sh_script_ps_out) > 0:
        return sh_script_ps_out[0].split()[3]
    else:
        return None


def run_experiment(hyper_params, nohup=True, prefix=None, init_greece_time=10, unit='GPU'):
    # Due to Theano+CUDA shitty initialization ->
    # run experiment as a separate proces
    ts = str(current_milli_time())
    if prefix is None:
        prefix = '{}-{}'.format(hyper_params['alg'], hyper_params['dataset'])
    os.makedirs('logs/', exist_ok=True)
    param_file_name = 'logs/param_{}_{}.json'.format(prefix, ts)
    with open(param_file_name, 'tw') as f:
        json.dump(hyper_params, f, sort_keys=True, indent=2)
    if nohup:
        os.system(
            "nohup sh run.sh experiments/experiment.py {} {} >logs/nohup_{}_{}.log 2>&1&".format(
                unit, param_file_name, prefix, ts
            )
        )
    else:
        os.system("sh run.sh experiments/experiment.py {} {}".format(unit, param_file_name))

    print("Waiting for process init {} seconds...".format(init_greece_time))
    time.sleep(init_greece_time)
    pid = get_pid_for_experiment_parma_file(param_file_name)
    return param_file_name, pid


ALL_GPU = 2
SLEEP_MIN = 3


def run_experiments_with_throttling(
    hyper_param_list, gpu=True, max_occupied_gpus=ALL_GPU, nohup=True, max_processes=ALL_GPU
):
    running_proc = dict()
    for idx, hp in enumerate(hyper_param_list):
        # check if maximum number of processes is not reached
        while True:
            print('Checking which processes still running...')
            for pf in list(running_proc.keys()):
                pf_pid = get_pid_for_experiment_parma_file(pf)
                print('Param file:', pf, 'PID:', pf_pid)
                if pf_pid is None:
                    del running_proc[pf]

            if len(running_proc) < max_processes:
                break
            print(
                "Max number of proc: {} reached. Waiting for {} min for other processes to end...".format(
                    max_processes, SLEEP_MIN
                )
            )
            time.sleep(SLEEP_MIN * 60)

        # # check available gpu
        # while gpu:
        #     working_gpus = occupied_gpus()
        #     print("Working GPU: ", working_gpus)
        #     if len(working_gpus) < max_occupied_gpus:
        #         break
        #     print(
        #         "Max number of occupied GPU: {} reached. Waiting for {} min for free GPU...".format(
        #             max_occupied_gpus, SLEEP_MIN
        #         )
        #     )
        #     time.sleep(SLEEP_MIN * 60)

        print("Running sample: {}/{}".format(idx + 1, len(hyper_param_list)))
        print(json.dumps(hp))
        pf, pid = run_experiment(hp, unit='GPU' if gpu else 'cpu', nohup=nohup)
        print('Param file:', pf, 'PID:', pid)
        running_proc[pf] = pid


def run_with_hyper_params_search(
    hyper_params_dist,
    random_seach=False,
    samples_num=None,
    gpu=True,
    nohup=True,
    max_occupied_gpus=ALL_GPU,
    max_processes=ALL_GPU
):
    assert max_occupied_gpus <= ALL_GPU
    hp_space = np.array([len(v) for v in hyper_params_dist.values()]).prod()
    print("All hyper-params combinations:", hp_space)

    if random_seach:
        assert samples_num is not None
        print("Random search with {} samples.".format(samples_num))
        samples = ParameterSampler(hyper_params_dist, samples_num)
    else:
        print("Grid search.")
        samples_num = hp_space
        samples = ParameterGrid(hyper_params_dist)

    run_experiments_with_throttling(samples, gpu, max_occupied_gpus, nohup, max_processes)
