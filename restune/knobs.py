import json
import time
import pandas as pd
import bisect
import numpy as np
from .utils import logger
from .utils.parser import  convert_65IM_to_51IM
import numpy as np
import os
import re
import ast
ts = int(time.time())
logger = logger.get_logger('autotune', 'log/train_ddpg_{}.log'.format(ts))
INTERNAL_METRICS_LEN = 51
# Deprecated Var Definition
KNOBS = [
         'innodb_max_dirty_pages_pct',
         'innodb_io_capacity',
         'innodb_max_dirty_pages_pct_lwm',
         'innodb_thread_concurrency',
         'innodb_lock_wait_timeout',
         'innodb_lru_scan_depth',
         'innodb_buffer_pool_instances',
         'innodb_purge_threads',
         'innodb_read_io_threads',
         'innodb_write_io_threads',
         'innodb_spin_wait_delay',
         'table_open_cache',
         'binlog_cache_size',
         'innodb_adaptive_max_sleep_delay',
         'innodb_change_buffer_max_size',
         'innodb_flush_log_at_timeout',
         'innodb_flushing_avg_loops',
         'innodb_max_purge_lag',
         'innodb_read_ahead_threshold',
         'innodb_sync_array_size',
         'innodb_sync_spin_loops',
         'metadata_locks_hash_instances',
         'tmp_table_size',
         'table_open_cache_instances',
         'thread_cache_size',
         'innodb_adaptive_hash_index_parts',
         'innodb_page_cleaners',
         'innodb_flush_neighbors',
]
KNOB_DETAILS = None
EXTENDED_KNOBS = None
num_knobs = len(KNOBS)


def gen_continuous(action):
    knobs = {}

    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]

        knob_type = value['type']

        if knob_type == 'integer':
            min_val, max_val = value['min'], value['max']
            delta = int((max_val - min_val) * action[idx])
            eval_value = min_val + delta 
            eval_value = max(eval_value, min_val)
            if value.get('stride'):
                all_vals = np.arange(min_val, max_val, value['stride'])
                indx = bisect.bisect_left(all_vals, eval_value)
                if indx == len(all_vals): indx -= 1
                eval_value = all_vals[indx]
            # TODO(Hong): add restriction among knobs, truncate, etc
            knobs[name] = eval_value
        if knob_type == 'float':
            min_val, max_val = value['min'], value['max']
            delta = (max_val - min_val) * action[idx]
            eval_value = min_val + delta
            eval_value = max(eval_value, min_val)
            all_vals = np.arange(min_val, max_val, value['stride'])
            indx = bisect.bisect_left(all_vals, eval_value)
            if indx == len(all_vals): indx -= 1
            eval_value = all_vals[indx]
            knobs[name] = eval_value
        elif knob_type == 'enum':
            enum_size = len(value['enum_values'])
            enum_index = int(enum_size * action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = value['enum_values'][enum_index]
            # TODO(Hong): add restriction among knobs, truncate, etc
            knobs[name] = eval_value
        elif knob_type == 'combination':
            enum_size = len(value['combination_values'])
            enum_index = int(enum_size * action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = value['combination_values'][enum_index]
            knobs_names = name.strip().split('|')
            knobs_value = eval_value.strip().split('|')
            for k, knob_name_tmp in enumerate(knobs_names):
                knobs[knob_name_tmp] = knobs_value[k]


    return knobs


def save_knobs(knobs, external_metrics):
    knob_json = json.dumps(knobs)
    result_str = '{},{},{},'.format(external_metrics[0], external_metrics[1], external_metrics[2])
    result_str += knob_json


def initialize_knobs(knobs_config):
    global KNOBS
    global KNOB_DETAILS
    f = open(knobs_config)
    KNOB_DETAILS = json.load(f)
    KNOBS = list(KNOB_DETAILS.keys())
    f.close()
    return KNOB_DETAILS


def get_default_knobs():
    default_knobs = {}
    for name, value in KNOB_DETAILS.items():
        if not value['type'] == "combination":
            default_knobs[name] = value['default']
        else:
            knobL = name.strip().split('|')
            valueL = value['default'].strip().split('|')
            for i in range(0, len(knobL)):
                default_knobs[knobL[i]] = int(valueL[i])
    return default_knobs


def get_knob_details(knobs_config):
    initialize_knobs(knobs_config)
    return KNOB_DETAILS


def knob2action(knob):
    actionL = []
    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]
        #knob_type = value['type']
        min_val, max_val = value['min'], value['max']
        action = (knob[name] - min_val) / (max_val - min_val)
        actionL.append(action)

    return np.array(actionL)

def knobDF2action(df):
    actionL = pd.DataFrame()
    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]
        if value['type'] in  ["integer", "float"]:
            min_val, max_val = value['min'], value['max']
            action = (df[name] - min_val) / (max_val - min_val)
            actionL[name] = action
        if value['type'] ==  "enum":
            actionL[name]=''
            enum_size = len(value['enum_values'])
            for i in range(0, df.shape[0]):
                actionL[name].iloc[i] = value['enum_values'].index(df[name].iloc[i]) / enum_size
        if value['type'] == "combination":
            actionL[name] = ''
            combination_size = len(value['combination_values'])
            combination_knobs = name.strip().split('|')
            for i in range(0, df.shape[0]):
                combination_value = ""
                for knob in combination_knobs:
                    if combination_value == "":
                        combination_value = str(df[knob].iloc[i])
                    else:
                        combination_value = combination_value + "|" + str(df[knob].iloc[i])
                actionL[name].iloc[i] = value['combination_values'].index(combination_value) / combination_size

    return np.array(actionL)


def get_data_for_mapping(fn):
    '''
    get konbs and tps from res file.
    Only those that meet the JSON requirements are returned. If none suitable, return FALSE
    '''
    f = open(fn)
    lines = f.readlines()
    f.close()
    konbL = []
    tpsL = []
    internal_metricL = []
    if os.path.getsize(fn) == 0:
        return False

    #check whether KNOBS is contained in the res file, for further mapping
    line = lines[0]
    t = line.strip().split('|')
    knob_str = t[0]
    knob_str = re.sub(r"[A-Z]+", "1", knob_str)
    tmp = re.findall(r"\D*", knob_str)
    old_knob = [name.strip('_') for name in tmp if name not in ['', '.']]
    combinationL = []

    for knob_name in KNOBS:
        if '|' in knob_name: #deal with combination type
            knobs = knob_name.split('|')
            combinationL.append(knob_name)
            for knob in knobs:
                if not knob in old_knob:
                    return False
        else:
            if not knob_name in old_knob:
                return False

    for line in lines:
        t = line.strip().split('|')
        knob_str = t[0]
        valueL_tmp = re.findall('(\d+(?:\.\d+)?)', knob_str)
        valueL = []
        for item in valueL_tmp:
            if item.isdigit():
                valueL.append(int(item))
            else:
                try:
                    valueL.append(float(item))
                except:
                    valueL.append(item)
        knob_str = re.sub(r"[A-Z]+", "1", knob_str)
        tmp = re.findall(r"\D*", knob_str)
        nameL = [name.strip('_') for name in tmp if name not in ['', '.']]
        tps = float(t[1])

        internal_metric = ast.literal_eval(t[4])
        if not len(internal_metric) == INTERNAL_METRICS_LEN:
            if len(t) == 5:
                del(internal_metric[51:65])
            elif t[5] == '65d':
                internal_metric = list(convert_65IM_to_51IM(np.array(internal_metric)))

        if len(combinationL):
            for name in combinationL:
                value = KNOB_DETAILS[name]
                combination_knobs = name.strip().split('|')
                combination_value = ""
                for knob in combination_knobs:
                    if combination_value == "":
                        combination_value = str(valueL[nameL.index(knob)])
                    else:
                        combination_value = combination_value + "|" + str(valueL[nameL.index(knob)])

                if not combination_value in value['combination_values']:
                    # the combination value is not in the range appointed in json, abort that row
                    continue

        konbL.append(valueL)
        tpsL.append(tps)
        internal_metricL.append(internal_metric)


    if len(tpsL) ==0:
        return False

    knob_df = pd.DataFrame(konbL, columns=nameL)
    internal_metricM = np.vstack(internal_metricL)

    #logger.info("51 metrics: {}".format(fn))
    return knob_df, tpsL, internal_metricM

