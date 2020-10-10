import os
import time
import pickle
import subprocess
import numpy as np
from .knobs import gen_continuous
from .dbenv import RESTART_WAIT_TIME, FixKnobsSimulatorEnv
from .knobs import logger
from .knobs import ts, knobDF2action, get_data_for_mapping
from .gp_tf import get_action_data, get_action_gp, get_action_data_from_res, get_best_action_gp, get_pred_gp
from .gp_torch import initialize_GP_model, anlytic_optimize_acqf_and_get_observation, get_acqf
from botorch import fit_gpytorch_model
from .WGPE import initialize_WGPE_model, optimize_acqf_and_get_observation_wgpe, get_acqf_wgpe
import torch
import pdb
from .utils.autotune_exceptions import AutotuneError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .utils.binner import Bin
from .utils.parser import convert_65IM_to_51IM, get_action_data_from_res_cpu
from .gp import gp_predict
RESTART_FREQUENCY = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def save_state_actions(state_action, filename):
    f = open(filename, 'wb')
    pickle.dump(state_action, f)
    f.close()


def generate_knobs(action, method):
    if method in ['ddpg', 'ppo', 'sac', 'gp']:
        return gen_continuous(action)
    else:
        raise NotImplementedError("Not implemented generate_knobs")


class MySQLTuner:
    def __init__(self, model, env, batch_size, episodes,
                 replay_memory='', idx_sql='', source_data_path='', dst_data_path='', method='DDPG', rl_log='', lhs_log='', restore_state='', workload_map='false', output_log='../lab/output_log', gp_model_dir='gp_model'):
        self.model = model
        self.env = env
        self.batch_size = batch_size
        self.episodes = episodes
        if replay_memory:
            self.model.replay_memory.load_memory(replay_memory)
            logger.info('Load memory: {}'.format(self.model.replay_memory))
        self.idx_sql = idx_sql
        self.src = source_data_path
        self.dst = dst_data_path
        self.fine_state_actions = []
        self.train_step = 0
        self.accumulate_loss = [0, 0]
        self.step_counter = 0
        self.expr_name = 'train_{}'.format(ts)
        # ouprocess
        self.sigma = 0.2
        # decay rate
        self.sigma_decay_rate = 0.99
        MySQLTuner.create_output_folders()
        # time for every step, time for training
        self.step_times, self.train_step_times = [], []
        # time for setup + restart + test, time for restart, choose action time
        self.env_step_times, self.env_restart_times, self.action_step_times = [], [], []
        self.noisy = False
        self.reinit = True
        if source_data_path == '' and dst_data_path == '':
            self.reinit = False
        if self.env.rds_mode:
            self.reinit = False
        self.method=method
        self.rl_log = rl_log
        self.lhs_log = lhs_log
        self.restore_state = restore_state
        self.workload_map = workload_map
        self.output_log = output_log
        self.gp_model_dir = gp_model_dir

    @staticmethod
    def create_output_folders():
        output_folders = ['log', 'save_memory', 'save_knobs', 'save_state_actions', 'model_params']
        for folder in output_folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

    def tune_GP(self):
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'

        f = open(fn, 'a')
        internal_metrics, initial_metrics = self.env.initialize()
        logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
            initial_metrics[0], initial_metrics[1], initial_metrics[2]
        ))
        #record initial data in res
        res = '{}|{}|{}|{}|{}|65d\n'.format(FixKnobsSimulatorEnv.stringify(self.env.default_knobs), str(initial_metrics[0]), str(initial_metrics[1]),
                                        str(initial_metrics[2]), list(internal_metrics))
        logger.info("{} is recorded in {}".format(res, fn))
        f.write(res)
        f.close()
        internalm_matrix = None
        if self.rl_log != '':
            action_df, tps = get_action_data(self.rl_log)
        elif self.lhs_log != '':
            if self.workload_map:
                action_df, tps, internalm_matrix = get_data_for_mapping(self.lhs_log)
            else:
                action_df, tps = get_action_data_from_res(self.lhs_log)
            action_df = knobDF2action(action_df) # normalize
        else:
            raise AutotuneError('no initial data provided for GP')
        f = open(fn, 'a')
        record_num = len(tps)
        X_scaled = action_df[:record_num, :]

        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))
        for global_step in range(10000):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                _, initial_metrics = self.env.initialize()
                logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
                    initial_metrics[0], initial_metrics[1], initial_metrics[2]
                ))
                latest_db_size = self.env.get_db_size()
                if latest_db_size > 4 * db_size:  # 4 times larger than it's original size
                    logger.warning('[Database Size Warning]. Your database {} size now is {}. We recommend you to restart your training task!'.format(self.env.dbname, latest_db_size))
                else:
                    logger.info('[Database Size Warning]. Your database {} size now is {}. You are all good.'.format(self.env.dbname, latest_db_size))

            normalized = StandardScaler()
            y_scaled = normalized.fit_transform(np.array(tps).reshape(-1, 1))
            y_scaled = -y_scaled

            if global_step % 10 == 0:
                action, ypreds, sigmas, queue_generated, best_tps_in_queue = get_best_action_gp(X_scaled, y_scaled)
                logger.info('[gp] Best Action: {}'.format(action))
                if queue_generated:
                    best_tps_in_queue = normalized.inverse_transform((-best_tps_in_queue).reshape(1))
                    tps_pred = normalized.inverse_transform(-ypreds)
                    tps_std = normalized.inverse_transform(-ypreds + np.sqrt(sigmas)) - tps_pred
                    logger.info('[gp][Episode: 1][Step: {}]: tps_pred: {}, tps_std: {}, tps in queue: {}'.format(global_step, tps_pred[0], tps_std[0], best_tps_in_queue[0]))
                    logger.info("[gp] Best Action is from the queue, no need to run the benchmark again")
                    continue
            else:
                if self.workload_map:
                    # matched_X_scaled, matched_tps: matched workload
                    matched_workload = self.map_workload(X_scaled, internalm_matrix)
                    matched_action_df, matched_tps = get_action_data_from_res(
                        '{}/{}'.format(self.output_log, matched_workload))
                    matched_action_df = knobDF2action(matched_action_df)  # normalize

                    record_num = len(matched_tps)
                    matched_X_scaled = matched_action_df[:record_num, :]

                    #normalized = StandardScaler()
                    matched_tps = matched_tps + tps
                    matched_y_scaled = normalized.fit_transform(np.array(matched_tps).reshape(-1, 1))
                    matched_y_scaled = -matched_y_scaled
                    matched_X_scaled = np.vstack((matched_X_scaled, X_scaled))
                    action, ypreds, sigmas = get_action_gp(matched_X_scaled, matched_y_scaled)
                    logger.info('[gp] Action: {}'.format(action))


                else:
                    action, ypreds, sigmas = get_action_gp(X_scaled, y_scaled)
                    logger.info('[gp] Action: {}'.format(action))

            tps_pred = normalized.inverse_transform(-ypreds)
            tps_std = normalized.inverse_transform(-ypreds + np.sqrt(sigmas)) - tps_pred
            logger.info('[gp][Episode: 1][Step: {}]: tps_pred: {}, tps_std: {}'.format(global_step, tps_pred[0], tps_std[0]))
            X_scaled = np.vstack((X_scaled, action.reshape(1, X_scaled.shape[1])))

            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))

            metrics, internal_metrics, avg_cpu_usage = self.env.step_GP(current_knob, global_step)
            logger.info('[gp][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{}]'.format(
                global_step, metrics[0], metrics[1], metrics[2]))
            tps.append(metrics[0])  # tps is used for collecting dependent variables(Y) for GP
            if internal_metrics is not None:
                internal_metrics_tmp = convert_65IM_to_51IM(internal_metrics)
                internalm_matrix = np.vstack(
                    (internalm_matrix, internal_metrics_tmp.reshape(1, internalm_matrix.shape[1])))
            res = '{}|{}|{}|{}|{}|65d\n'.format(FixKnobsSimulatorEnv.stringify(current_knob), str(metrics[0]), str(metrics[1]),str(metrics[2]), list(internal_metrics))
            logger.info("{} is recorded in {}".format(res, fn))
            f.write(res)

            # TODO with enough data, no workload mapping
            if len(tps) >= 50:
                self.workload_map = False

        f.close()
        return
    
    def tune_GP_Botorch(self):
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        f = open(fn, 'a')
        internal_metrics, initial_metrics = self.env.initialize()
        logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
            initial_metrics[0], initial_metrics[1], initial_metrics[2]
        ))
        #record initial data in res
        res = '{}|{}|{}|{}|{}|65d\n'.format(FixKnobsSimulatorEnv.stringify(self.env.default_knobs), str(initial_metrics[0]), str(initial_metrics[1]),
                                        str(initial_metrics[2]), list(internal_metrics))
        logger.info("{} is recorded in {}".format(res, fn))
        f.write(res)
        internalm_matrix = None
        if self.rl_log != '':
            action_df, tps = get_action_data(self.rl_log)
        elif self.lhs_log != '':
            if self.workload_map:
                action_df, tps, internalm_matrix = get_data_for_mapping(
                    self.lhs_log)
            else:
                action_df, tps = get_action_data_from_res(self.lhs_log)
            action_df = knobDF2action(action_df)  # normalize
        else:
            raise AutotuneError('no initial data provided for GP')
        record_num = len(tps)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.float64)
        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))
        NUM_STEP = 10000
        # set acquisition func
        acqf_name = 'EI'
        cadidate_size = 1
        reusable = False
        normalized = StandardScaler()
        for global_step in range(NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                _, initial_metrics = self.env.initialize()
                logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
                    initial_metrics[0], initial_metrics[1], initial_metrics[2]
                ))
                latest_db_size = self.env.get_db_size()
                if latest_db_size > 4 * db_size:  # 4 times larger than it's original size
                    logger.warning('[Database Size Warning]. Your database {} size now is {}. We recommend you to restart your training task!'.format(
                        self.env.dbname, latest_db_size))
                else:
                    logger.info('[Database Size Warning]. Your database {} size now is {}. You are all good.'.format(
                        self.env.dbname, latest_db_size))

            if self.workload_map:
                # matched_X_scaled, matched_tps: matched workload
                matched_workload = self.map_workload(
                    X_scaled, internalm_matrix)
                matched_action_df, matched_tps = get_action_data_from_res(
                    '{}/{}'.format(self.output_log, matched_workload))
                matched_action_df = knobDF2action(
                    matched_action_df)  # normalize

                record_num = len(matched_tps)
                matched_X_scaled = matched_action_df[:record_num, :]

                #normalized = StandardScaler()
                matched_tps = matched_tps + tps
                matched_y_scaled = normalized.fit_transform(
                    np.array(matched_tps).reshape(-1, 1))
                matched_X_scaled = np.vstack((matched_X_scaled, X_scaled))
                action, ypreds, sigmas = get_action_gp(matched_X_scaled, matched_y_scaled)
                logger.info('[GP-BOTORCH] Action: {}'.format(action))
                train_x = torch.tensor(matched_X_scaled)
                train_obj = torch.tensor(matched_y_scaled)
            else:
                y_scaled = normalized.fit_transform(
                    np.array(tps).reshape(-1, 1))
                train_x = torch.tensor(X_scaled)
                train_obj = torch.tensor(y_scaled)

            if reusable and model:
                mll, model = initialize_GP_model(
                    train_x,
                    train_obj,
                    model.state_dict()
                )
            else:
                mll, model = initialize_GP_model(train_x, train_obj)
                fit_gpytorch_model(mll)


            acqf = get_acqf(acqf_name, model, train_obj)
            bounds = torch.tensor([[0.0] * X_scaled.shape[1], [1.0] * X_scaled.shape[1]], device=device, dtype=torch.double)
            if acqf is None:
                AutotuneError("acqf none")
            new_x = anlytic_optimize_acqf_and_get_observation(acqf, cadidate_size, bounds)
            # update training points
            action = new_x.squeeze(0).numpy()
            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))
            metrics, internal_metrics, avg_cpu_usage = self.env.step_GP(current_knob, global_step)
            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                global_step, metrics[0], metrics[1], metrics[2], avg_cpu_usage))
            tps.append(metrics[0])
            X_scaled = np.vstack((X_scaled, new_x.numpy()))
            logger.info('[GP-BOTORCH] Action: {}'.format(action))

            if internalm_matrix is not None:
                internal_metrics_tmp = convert_65IM_to_51IM(internal_metrics)
                internalm_matrix = np.vstack((internalm_matrix, internal_metrics_tmp.reshape(1, internalm_matrix.shape[1])))
            res = '{}|{}|{}|{}|{}|65d\n'.format(FixKnobsSimulatorEnv.stringify(current_knob), str(metrics[0]), str(metrics[1]), str(metrics[2]), list(internal_metrics))
            logger.info("{} is recorded in {}".format(res, fn))
            f.write(res)

            # TODO with enough data, no workload mapping
            if len(tps) >= 50:
                self.workload_map = False

        f.close()
        return

    def tune_FixKnobs(self):
        repeat_time = 2
        action_df, tps = get_action_data_from_res(self.lhs_log)
        tps_sum = {}
        for i in range(1, repeat_time+1):
            if i%2 == 1:
                index = range(0, action_df.shape[0])
            else:
                index = reversed(range(0, action_df.shape[0]))
            for j in index:
                current_knob = action_df.iloc[j].to_dict()
                logger.info('knobs generated: {}'.format(current_knob))
                metrics, internal_metrics, _ = self.env.step_GP(current_knob, j)
                if not j in tps_sum.keys():
                    tps_sum[j] = metrics[0]
                else:
                    tps_sum[j] = tps_sum[j] + metrics[0]
                logger.info('[gp][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{}]'.format(
                    j, metrics[0], metrics[1], metrics[2]))
                logger.info('[gp][Episode: 1][Step: {}][Metric average tps:{}  original tps:{}'.format(
                    j, tps_sum[j]/i, tps[j]))

    def tune_WGPE(self):
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        f = open(fn, 'a')
        internal_metrics, initial_metrics = self.env.initialize()
        logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
            initial_metrics[0], initial_metrics[1], initial_metrics[2]
        ))
        #record initial data in res
        res = '{}|{}|{}|{}|{}|65d\n'.format(FixKnobsSimulatorEnv.stringify(self.env.default_knobs), str(initial_metrics[0]), str(initial_metrics[1]),
                                        str(initial_metrics[2]), list(internal_metrics))
        logger.info("{} is recorded in {}".format(res, fn))
        f.write(res)
        if self.rl_log != '':
            action_df, tps = get_action_data(self.rl_log)
        elif self.lhs_log != '':
            action_df, tps = get_action_data_from_res(self.lhs_log)
            action_df = knobDF2action(action_df)  # normalize
        else:
            raise AutotuneError('no initial data provided for GP')
        record_num = len(tps)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.float64)
        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))
        NUM_STEP = 10000
        # set acquisition func
        for global_step in range(NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                _, initial_metrics = self.env.initialize()
                logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
                    initial_metrics[0], initial_metrics[1], initial_metrics[2]
                ))
                latest_db_size = self.env.get_db_size()
                if latest_db_size > 4 * db_size:  # 4 times larger than it's original size
                    logger.warning('[Database Size Warning]. Your database {} size now is {}. We recommend you to restart your training task!'.format(
                        self.env.dbname, latest_db_size))
                else:
                    logger.info('[Database Size Warning]. Your database {} size now is {}. You are all good.'.format(
                        self.env.dbname, latest_db_size))

            train_x = torch.tensor(X_scaled)
            train_obj = torch.tensor(tps).reshape(-1, 1)
            start = time.time()
            model = initialize_WGPE_model(train_x, train_obj, self.gp_model_dir)
            acq_fun = get_acqf_wgpe("EI", model, train_x, train_obj)
            new_x = optimize_acqf_and_get_observation_wgpe(acq_fun, train_x)
            logger.info('[WGPE] using {} to get new_x'.format(time.time()-start))
            # update training points
            action = new_x.squeeze(0).numpy()
            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))
            metrics, internal_metrics, _ = self.env.step_GP(current_knob, global_step)
            logger.info('[WGPE][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{}]'.format(
                global_step, metrics[0], metrics[1], metrics[2]))
            tps.append(metrics[0])
            X_scaled = np.vstack((X_scaled, new_x.numpy()))
            logger.info('[WGPE] Action: {}'.format(action))
            res = '{}|{}|{}|{}|{}|65d\n'.format(FixKnobsSimulatorEnv.stringify(current_knob), str(metrics[0]), str(metrics[1]), str(metrics[2]), list(internal_metrics))
            logger.info("{} is recorded in {}".format(res, fn))
            f.write(res)

        f.close()
        return


    def tune(self):
        if self.method=='GP_BOTORCH':
            return self.tune_GP_Botorch()
        elif self.method == 'WGPE':
            return self.tune_WGPE()

    def map_workload(self, target_X_scaled, y_target):
        # load old workload data
        workload_list = os.listdir(self.output_log)
        for f in workload_list:
            fn = os.path.join(self.output_log, f)
            if os.path.isdir(fn):
                workload_list.remove(f)
                continue
        workload_list.sort()
        # for matching the history internal metrics which have 65 dimensiona
        #y_target = convert_65IM_to_51IM(y_target)
        # vesion1: workload mapping by predicting TPS
        '''  
        # normalize
        normalizer = StandardScaler()
        target_tps_scaled = normalizer.fit_transform(np.array(target_tps).reshape(-1, 1))
        target_tps_scaled = - target_tps_scaled
        scores = {}
        for workload_id, workload in enumerate(workload_list):
            # load old workload data
            old_X, old_tps = get_action_data_from_res('{}/{}'.format(self.output_log, workload))
            old_X_scaled = knobDF2action(old_X)
            old_record_num = len(old_tps)
            old_X_scaled = old_X_scaled[:old_record_num, :]

            normalizer = StandardScaler()
            old_tps_scaled = normalizer.fit_transform(np.array(old_tps).reshape(-1, 1))
            old_tps_scaled = - old_tps_scaled

            # predict tps at target_action_df
            tps_pred = get_pred_gp(old_X_scaled, old_tps_scaled, target_X_scaled)
            dists = np.sqrt(np.sum(np.square(
                np.subtract(tps_pred, target_tps_scaled)), axis=1))
            scores[workload] = np.mean(dists)
        '''
        #version2: workload mapping by predicting internal metrics
        #obtain all data for mapping
        workload_dir = {}
        for workload_name in workload_list:
            # load old workload data
            if os.path.getsize(os.path.join(self.output_log, workload_name)) == 0:
                logger.info(('[Wokrload Mapping] {} is empty'.format(workload_name)))
                continue

            valid = get_action_data_from_res_cpu(os.path.join(self.output_log, workload_name))
            if not valid:
                continue
            old_X, old_tps, _, old_metrics = valid
            workload_dir[workload_name] = {}
            workload_dir[workload_name]['X_matrix'] = knobDF2action(old_X)
            workload_dir[workload_name]['y_matrix'] = old_metrics

        # Stack all y matrices for preprocessing
        ys = np.vstack([entry['y_matrix'] for entry in list(workload_dir.values())])
        # Scale the  y values, then compute the deciles for each column in y
        y_scaler = StandardScaler()
        y_scaler.fit_transform(ys)
        y_binner = Bin(bin_start=1)
        y_binner.fit(ys)
        del ys

        # Now standardize the target's data and bin it by the deciles we just calculated
        y_target = y_scaler.transform(y_target)
        y_target = y_binner.transform(y_target)

        # workload mapping by predicting internal metrics
        scores = {}
        for workload_id, workload_entry in list(workload_dir.items()):
            predictions = np.empty_like(y_target)
            X_scaled = workload_entry['X_matrix']
            y_workload = workload_entry['y_matrix']
            y_scaled = y_scaler.transform(y_workload)
            for j, y_col in enumerate(y_scaled.T):
                # Using this workload's data, train a Gaussian process model
                # and then predict the performance of each metric for each of
                # the knob configurations attempted so far by the target.
                y_col = y_col.reshape(-1, 1)
                predictions[:, j] = gp_predict(X_scaled, y_col, target_X_scaled, workload_id, j)
                # Bin each of the predicted metric columns by deciles and then
                # compute the score (i.e., distance) between the target workload
                # and each of the known workloads
            predictions = y_binner.transform(predictions)
            dists = np.sqrt(np.sum(np.square(
                np.subtract(predictions, y_target)), axis=1))
            scores[workload_id] = np.mean(dists)

        # Find the best (minimum) score
        best_score = np.inf
        best_workload = None
        for workload, similarity_score in list(scores.items()):
            if similarity_score < best_score:
                best_score = similarity_score
                best_workload = workload

        logger.info('[Wokrload Mapping] Score:{}'.format(str(scores)))
        logger.info('[Workload Mapping] Matched Workload: {}'.format(best_workload))

        return best_workload

