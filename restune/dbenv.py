import os
import re
import time
import math
import threading
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict

from .dbconnector import MysqlConnector
from .knobs import logger
from .utils.parser import ConfigParser
from .utils.parser import parse_tpcc
from .utils.resource_monitor import ResourceMonitor
from .knobs import initialize_knobs, save_knobs, get_default_knobs, knob2action

RETRY_WAIT_TIME = 10
RESTART_WAIT_TIME = 10
BENCHMARK_RUNNING_TIME = 30
DATABASE_REINIT_TIME = 600
BENCHMARK_WARMING_TIME= 0
REBALANCE_FREQUENCY = 5


class DatabaseType(Enum):
    Mysql = 1
    Postgresql = 2


class DBEnv(ABC):
    def __init__(self, workload):
        self.score = 0.
        self.steps = 0
        self.terminate = False
        self.workload = workload

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self, knobs, step):
        pass

    @abstractmethod
    def run(self):
        pass

    # TODO(Hong)
    @abstractmethod
    def terminate(self):
        return False


class MySQLEnv(DBEnv):
    def __init__(self,
                 workload,
                 knobs_config,
                 num_metrics,
                 log_path='',
                 threads=8,
                 host='localhost',
                 port=3392,
                 user='root',
                 passwd='',
                 dbname='tpcc',
                 constraint = 0):
        super().__init__(workload)
        self.knobs_config = knobs_config
        self.workload = workload
        self.log_path = log_path
        self.num_metrics = num_metrics
        self.default_external_metrics = []
        self.last_external_metrics = []
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.threads = threads
        self.best_result = './autotune_best.res'
        initialize_knobs(knobs_config)
        self.default_knobs = get_default_knobs()
        self.step_count = 0
        self.rm = ResourceMonitor(1, 120)   # hardcoded for 120 seconds
        self.constraint = constraint

    def get_external_metrics(self, filename=''):
        if self.workload['name'] == 'tpcc':
            result = parse_tpcc(filename)

    def get_resource_metrics(self, s_t, e_t):
        return self.rds_assist.get_rds_cpu_usage(s_t, e_t)

        def get_internal_metrics(self, internal_metrics):
        """Get the all internal metrics of MySQL, like io_read, physical_read.

        This func uses a SQL statement to lookup system table: information_schema.INNODB_METRICS
        and returns the lookup result.
        """
        _counter = 0
        _period = 5
        count = (BENCHMARK_RUNNING_TIME+BENCHMARK_WARMING_TIME) / _period -1
        warmup = BENCHMARK_WARMING_TIME/ _period

        def collect_metric(counter):
            counter += 1
            timer = threading.Timer(float(_period), collect_metric, (counter,))
            timer.start()
            db_conn = MysqlConnector(host=self.host,
                                     port=self.port,
                                     user=self.user,
                                     passwd=self.passwd,
                                     name=self.dbname,
                                     socket=self.sock)
            if counter >= count:
                timer.cancel()
            try:
                if counter > warmup:

                    sql = 'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME'
                    res = db_conn.fetch_results(sql, json=False)
                    res_dict = {}
                    for (k, v) in res:
                        #if not k in BLACKLIST:
                        res_dict[k] = v
                    internal_metrics.append(res_dict)
            except Exception as err:
                logger.info('Get Metrics error: {}'.format(err))

        collect_metric(_counter)
        return internal_metrics

    def _post_handle(self, metrics):
        result = np.zeros(self.num_metrics)

        def do(metric_name, metric_values):
            metric_type = 'counter'
            if metric_name in value_type_metrics:
                metric_type = 'value'
            if metric_type == 'counter':
                return float(metric_values[-1] - metric_values[0])*23/len(metric_values)
            else:
                return float(sum(metric_values)) / len(metric_values)

        keys = list(metrics[0].keys())
        keys.sort()
        for idx in range(len(keys)):
            key = keys[idx]
            data = [x[key] for x in metrics]
            result[idx] = do(key, data)
        return result

    def get_states(self, collect_cpu_remote=0):
                timestamp = int(time.time())
        filename = self.log_path + '/{}.log'.format(timestamp)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        internal_metrics = []
        self.get_internal_metrics(internal_metrics)
        start_ts = int(round(time.time() * 1000)) + BENCHMARK_WARMING_TIME * 1000
        end_ts = start_ts + BENCHMARK_RUNNING_TIME * 1000
        if not collect_cpu_remote:
            self.rm.run_only_cpu()
        if self.workload['name'] == 'tpcc':
            t = time.time()
            cmd = self.workload['cmd'].format(dirname+'/cli/run_tpcc.sh',
                                              self.host,
                                              self.port,
                                              self.user,
                                              self.threads,
                                              BENCHMARK_WARMING_TIME,
                                              BENCHMARK_RUNNING_TIME,
                                              filename,
                                              self.dbname)
            logger.info('[DBG]. {}'.format(cmd))
            osrst = os.system(cmd)
            t = time.time() - t
            if osrst != 0 or t < 10:
                logger.error('get states failed.')
        if collect_cpu_remote:
            avg_cpu_usage = self.get_resource_metrics(start_ts, end_ts)
        else:
            monitor_data_dict = self.rm.get_monitor_data()
            avg_cpu_usage = sum(monitor_data_dict['cpu']) / len(monitor_data_dict['cpu'])
        external_metrics = self.get_external_metrics(filename)
        internal_metrics = self._post_handle(internal_metrics)
        logger.info('internal metrics: {}.'.format(list(internal_metrics)))
        return external_metrics, internal_metrics, avg_cpu_usage

    def initialize(self, collect_CPU_remote=0):
        self.score = 0.
        self.steps = 0
        self.terminate = False

        logger.info('[DBG]. default tuning knobs: {}'.format(self.default_knobs))
        if self.rds_mode:
            self.apply_knobs(self.default_knobs)
        else:
            self.apply_knobs(self.default_knobs)
        logger.info('[DBG]. apply default knobs done')

        external_metrics, internal_metrics, avg_cpu_usage = self.get_states(collect_CPU_remote)
        logger.info('[DBG]. get_state done: {}|{}'.format(external_metrics, internal_metrics))

        # TODO(HONG): check external_metrics[0]
        while external_metrics[0] == 0 or sum(internal_metrics) == 0:
            logger.info('retrying: sleep for {} seconds'.format(RETRY_WAIT_TIME))
            time.sleep(RETRY_WAIT_TIME)
            logger.info('try getting_states again')
            external_metrics, internal_metrics, avg_cpu_usage = self.get_states(collect_CPU_remote)
            logger.info('[DBG]. get_state done: {}|{}|{}'.format(external_metrics, internal_metrics, avg_cpu_usage))
        self.last_cpu = avg_cpu_usage
        self.default_cpu = avg_cpu_usage

        state = internal_metrics
        save_knobs(self.default_knobs, external_metrics)
        return state, external_metrics, avg_cpu_usage

    def step(self, knobs, global_step, collect_CPU_remote=0):
        flag = self.apply_knobs(knobs)
        if not flag:
            return [0, 0, 0]

        s = self.get_states(collect_CPU_remote)

        if s is None:
            return [0, 0, 0]

        external_metrics, internal_metrics, avg_cpu_usage = s

        while external_metrics[0] == 0 or sum(internal_metrics) == 0:
            logger.info('retrying because got invalid metrics. Sleep for {} seconds.'.format(RETRY_WAIT_TIME))
            time.sleep(RETRY_WAIT_TIME)
            logger.info('try get_states again')
            external_metrics, internal_metrics, avg_cpu_usage = self.get_states()
            logger.info('invalid metrics got again, {}|{}'.format(external_metrics, internal_metrics))

        return external_metrics, internal_metrics, avg_cpu_usage

    def apply_knobs(self, knobs):
        db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
        if 'innodb_io_capacity' in knobs.keys():
            self.set_param(db_conn, 'innodb_io_capacity_max', 2 * int(knobs['innodb_io_capacity']))
        for k, v in knobs.items():
           self.set_param(db_conn, k, v)
        for key in knobs.keys():
            self.set_param(db_conn, key, knobs[key])
        db_conn.close_db()
        return True

    def set_param(self, db_conn, k, v):
        sql = 'SHOW VARIABLES LIKE "{}";'.format(k)
        r = db_conn.fetch_results(sql)
        if r[0]['Value'] == 'ON':
            v0 = 1
        elif r[0]['Value'] == 'OFF':
            v0 = 0
        else:
            try:
                v0 = eval(r[0]['Value'])
            except:
                v0 = r[0]['Value'].strip()
        if v0 == v:
            return True

        sql = 'SET GLOBAL {}={}'.format(k, v)
        db_conn.execute(sql)

        while not self._check_apply(db_conn, k, v, v0):
            time.sleep(1)
        return True

