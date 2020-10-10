import re
import os
from shutil import copyfile
from dynaconf import settings
import numpy as np
import pandas as pd
import ast

class ConfigParser(object):

    def __init__(self, cnf):
        if not os.path.exists(cnf):
            return
        f = open(cnf)
        self._cnf = cnf
        self._knobs = {}
        for line in f:
            if line.strip().startswith('skip-external-locking') \
                    or line.strip().startswith('[') \
                    or line.strip().startswith('#') \
                    or line.strip() == '':
                pass
            else:
                try:
                    k, _, v = line.strip().split()
                    self._knobs[k] = v
                except:
                    print(line)   
        f.close()

    def replace(self, tmp='/tmp/mysql.cnf'):
        if settings.TMP_DIR:
            tmp = settings.TMP_DIR + '/mysql.cnf'
        f1 = open(self._cnf)
        f2 = open(tmp, 'w')
        for line in f1:
            tpl = line.strip().split()
            if len(tpl) < 1:
                f2.write(line)
            elif tpl[0] in self._knobs:
                tpl[2] = self._knobs[tpl[0]]
                f2.write('%s\t\t%s %s\n' % (tpl[0], tpl[1], tpl[2]))
            else:
                f2.write(line)
        f1.close()
        f2.close()
        copyfile(tmp, self._cnf)

    def set(self, k, v):
        self._knobs[k] = v

    def generate(self, output):
        f1 = open(self._cnf)
        f2 = open(output, 'w')
        current_knobs = {}
        for line in f1:
            tpl = line.strip().split()
            if len(tpl) < 1:
                f2.write(line)
            elif tpl[0] in self._knobs:
                tpl[2] = self._knobs[tpl[0]]
                f2.write('%s\t\t%s %s\n' % (tpl[0], tpl[1], tpl[2]))
                current_knobs[tpl[0]] = 0
            else:
                f2.write(line)
        for k, v in self._knobs.items():
            if k not in current_knobs:
                f2.write('%s\t\t= %s\n' % (k, v))
        f1.close()
        f2.close()


def parse_tpcc(file_path):
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(".*?trx: (\d+), 95%: (\d+.\d+), 99%: (\d+.\d+), max_rt:.*?")
    temporal = temporal_pattern.findall(lines)
    tps, latency, qps = 0, 0, 0
    for i in temporal:
        tps += float(i[0])
        latency += float(i[2])
    num_samples = len(temporal)
    if num_samples != 0:
        tps /= num_samples
        latency /= num_samples
        # interval
        tps /= 1
    return [tps, latency, tps]


def parse_sysbench(file_path):
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(
                "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)"
                " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
    temporal = temporal_pattern.findall(lines)
    tps, latency, qps = 0, 0, 0

    for i in temporal[-10:]:
        tps += float(i[0])
        latency += float(i[5])
        qps += float(i[1])
    num_samples = len(temporal[-10:])
    if num_samples != 0:
        tps /= num_samples
        qps /= num_samples
        latency /= num_samples
    else:
        print('num_samples is zero!')
    return [tps, latency, qps]


def parse_cloudbench(file_path):
    f = open(file_path)
    qps_list = []
    for line in f:
        if 'Request/s' in line:
            v = line.split()[6].split(':')[1]
            qps_list.append(float(v))
    qps = sum(qps_list) / float(len(qps_list))
    return [qps, 0, qps]


def parse_oltpbench(file_path):
    # file_path = *.summary
    with open(file_path) as f:
        lines = f.read()

    tps_temporal_pattern = re.compile("Throughput.*?(\d+.\d+),")
    tps_temporal = tps_temporal_pattern.findall(lines)
    tps = float(tps_temporal[0])

    lat_temporal_pattern = re.compile("95th.*?(\d+.\d+),")
    lat_temporal = lat_temporal_pattern.findall(lines)
    latency = float(lat_temporal[0])

    return [tps, latency, tps]

'''
temporary function: convert internal metrics of 65 dimensions to 51 dimensions
We need this function, because the history internal metrics we collected is 51 dimensions,
and we want to collect all internal metrics(65 dimensions) in later run. So, when mapping,
we convert the dimension for the purpose of matching  history data. 
'''
def convert_65IM_to_51IM(metrics):
    extra_metrics_index = [11, 12, 15, 40, 37, 43, 47, 48, 49, 50, 51, 62, 63, 64]
    if len(metrics.shape) == 2:
        if metrics.shape[1] == 51:
            return metrics
        metrics = np.delete(metrics, extra_metrics_index, axis=1)
    else:
        if metrics.shape[0] == 51:
            return metrics
        metrics = np.delete(metrics, extra_metrics_index)

    return metrics

def get_action_data_from_res_cpu(fn):
    f = open(fn)
    lines = f.readlines()
    f.close()
    metricsL = []
    tpsL, cpuL, internal_metricL = [], [], []
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
        cpu = float(t[4])
        if t[-1] == '65d':
            internal_metric = ast.literal_eval(t[-2])
        else:
            internal_metric = ast.literal_eval(t[-1])

        metricsL.append(valueL)
        tpsL.append(tps)
        cpuL.append(cpu)
        internal_metricL.append(internal_metric)
    df = pd.DataFrame(metricsL, columns=nameL)

    return df, tpsL, cpuL, np.vstack(internal_metricL)


def get_action_data_from_res_cpu2(fn):
    f = open(fn)
    lines = f.readlines()
    f.close()
    metricsL = []
    tpsL, cpuL, latencyL, internal_metricL = [], [], [], []
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
        cpu = float(t[4])
        latency = float(t[2])
        if t[-1] == '65d':
            internal_metric = ast.literal_eval(t[-2])
        else:
            internal_metric = ast.literal_eval(t[-1])

        metricsL.append(valueL)
        tpsL.append(tps)
        cpuL.append(cpu)
        latencyL.append(latency)
        internal_metricL.append(internal_metric)
    df = pd.DataFrame(metricsL, columns=nameL)

    return df, tpsL, cpuL, latencyL, np.vstack(internal_metricL)
