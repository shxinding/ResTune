import os
import argparse

from restune.dbenv import MySQLEnv
from restune.workload import TPCC_RDS_WORKLOAD
from restune.tuner_cpu import CPUTuner
from restune.knobs import logger
from restune.utils.helper import check_env_setting
from restune.utils.restune_exceptions import RestuneError

USER='root'
PASSWD=''
LOG_PATH='./log/'
PORT=3306
HOST='localhost'
THREADS=64

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='GP_BOTORCH', help='tunning method[GP_BOTORCH, WGPE]')
    parser.add_argument('--benchmark', type=str, default='tpcc', help='[sysbench, tpcc, workload_zoo, \
                        oltpbench_wikipedia, oltpbench_syntheticresourcestresser, oltpbench_twitter, oltpbench_tatp, \
                        oltpbench_auctionmark, oltpbench_seats, oltpbench_ycsb, oltpbench_jpab, \
                        oltpbench_ch-benchmark, oltpbench_voter, oltpbench_slbench, oltpbench_smallbank, oltpbench_linkbench]')
    parser.add_argument('--knobs_config', type=str, default='', help='knobs configuration file in json format')
    parser.add_argument('--lhs_log', type=str, default='', help='log file generated from lhs for GP traning,like xxx.res')
    parser.add_argument('--output_log', type=str, default='../lab/output_log', help='old workload data output_log')
    parser.add_argument('--tps_constraint', type=int, default=700, help='tps constraint')
    # wgpe
    parser.add_argument('--gp_cpu_model_dir', type=str, default='gp_model_cpu', help='source cpu model for WGPE')
    parser.add_argument('--gp_tps_model_dir', type=str, default='gp_model_tps', help='source tps model for WGPE')
    opt = parser.parse_args()

    if opt.knobs_config == '':
        err_msg = 'You must specify the knobs_config file for tuning: --knobs_config=knobs.json'
        logger.error(err_msg)
        raise RestuneError(err_msg)

    # model
    model = ''

    # Check env
    check_env_setting(opt.benchmark, True)

    # env
    wl = None
    dbname = opt.dbname
    if opt.benchmark == 'tpcc':
        wl = dict(TPCC_RDS_WORKLOAD)

    env = MySQLEnv(workload=wl,
                   knobs_config=opt.knobs_config,
                   num_metrics=65,
                   log_path=LOG_PATH,
                   threads=THREADS,
                   host=HOST,
                   port=PORT, # 3306 for rds
                   user=USER,
                   passwd=PASSWD,
                   dbname=dbname,
                   sock='',
                   constraint=opt.tps_constraint)
    logger.info('env initialized with the following options: {}'.format(opt))

    tuner = CPUTuner(model=model,
                     env=env,
                     batch_size=16,
                     episodes=100,
                     replay_memory='',
                     idx_sql='',
                     source_data_path='',
                     dst_data_path='',
                     method=opt.method,
                     lhs_log=opt.lhs_log,
                     tps_constrained=opt.tps_constraint,
                     gp_tps_model_dir=opt.gp_tps_model_dir,
                     gp_cpu_model_dir=opt.gp_cpu_model_dir,
                     output_log=opt.output_log)  # for TPCC200, RDS 15
    tuner.tune()
