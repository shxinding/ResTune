import os
import time
import pickle
import subprocess
import numpy as np
from .knobs import gen_continuous
from .dbenv import RESTART_WAIT_TIME
from .knobs import logger
from .knobs import ts, knobDF2action, get_data_for_mapping, knob2action
from .gp_tf import get_action_data, get_action_gp, get_action_data_from_res, get_best_action_gp, get_pred_gp
from .gp_torch import initialize_GP_model, anlytic_optimize_acqf_and_get_observation, get_acqf
from .WGPE import initialize_WGPE_model, optimize_acqf_and_get_observation_wgpe, get_acqf_wgpe
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.models import ModelListGP
import pdb
from .utils.autotune_exceptions import AutotuneError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from .SMAC import SMAC
#from smac.configspace import ConfigurationSpace
#from smac.facade.smac_hpo_facade import SMAC4HPO
#from smac.scenario.scenario import Scenario
#from smac.initial_design.default_configuration_design import DefaultConfiguration
from .utils.parser import convert_65IM_to_51IM, get_action_data_from_res_cpu
from .tuner import MySQLTuner, generate_knobs, save_state_actions
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf
from .WGPE_multiOutput import initialize_WGPE_model_multiOutput, initialize_WGPE_model_meta


RESTART_FREQUENCY = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

NUM_STEP=10000

def outcome_constraint(tps_real, tps_constrained):
    """L1 constraint; feasible if less than or equal to zero."""
    return tps_constrained - tps_real


def stringify(cnf):
    sorted_keys = sorted(cnf.keys())
    return ''.join(['_{}_{}'.format(key, str(cnf[key])) for key in sorted_keys])


class CPUTuner(MySQLTuner):
    def __init__(self, model, env, batch_size, episodes,
                 replay_memory='', idx_sql='', source_data_path='', dst_data_path='', method='DDPG', rl_log='', lhs_log='',
                 restore_state='', workload_map=False, gp_model_dir='', output_log='../lab/output_log', tps_constrained=1e9,
                 gp_tps_model_dir='gp_model_tps', gp_cpu_model_dir='gp_model_cpu', default_constraint=False):
        super().__init__( model, env, batch_size, episodes,
                 replay_memory, idx_sql, source_data_path, dst_data_path, method, rl_log, lhs_log,
                          restore_state, workload_map, output_log, gp_model_dir)
        self.tps_constrained = tps_constrained
        self.gp_tps_model_dir = gp_tps_model_dir
        self.gp_cpu_model_dir = gp_cpu_model_dir
        self.default_constraint = default_constraint

    def CEI_debug(self, cei, point, global_step):
        cei_b = cei(point).item()
        posterior = cei._get_posterior(X=point)
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)
        means = posterior.mean.squeeze(dim=-2)
        prob_b = cei._compute_prob_feas(X=point, means=means, sigmas=sigmas).item()
        ei_b = cei_b / (prob_b + 1e-9)
        logger.info(
            "[GP-BOTORCH][Episode: 1][Step: {}] CEI:{}, prob:{}, EI:{}, CPU_mean:{}, CPU_sigmas:{}, TPS_mean:{}, TPS_sigmas:{}".format(
                global_step, cei_b, prob_b, ei_b, means[1], sigmas[1], means[0], sigmas[0],
            ))

    def tune_GP_Botorch(self, collect_cpu_remote=True):
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        f = open(fn, 'a')
        internal_metrics, initial_metrics, resource = self.env.initialize(collect_cpu_remote)
        logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(resource,
            initial_metrics[0], initial_metrics[1], initial_metrics[2]))
        if self.default_constraint:
            self.tps_constrained = initial_metrics[0] * 0.95
            logger.info('[GP-BOTORCH][Episode: 1][Step: 0 tps_constraint: {}]'.format(self.tps_constrained))

        res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(self.env.default_knobs), str(initial_metrics[0]), str(initial_metrics[1]),
                                        str(initial_metrics[2]),resource, '', '', '', '', list(internal_metrics))

        logger.info("{} is recorded in {}".format(res, fn))
        f.write(res)
        f.close()
        f = open(fn, 'a')
        internalm_matrix = None
        if self.rl_log != '':
            action_df, tps = get_action_data(self.rl_log)
        elif self.lhs_log != '':
            if self.workload_map:
                action_df, tps, internalm_matrix = get_data_for_mapping(
                    self.lhs_log)
            else:
                action_df, tpsL, cpuL, _ = get_action_data_from_res_cpu(self.lhs_log)
                # cpuL = [-cpu for cpu in cpuL] #botorch assumes a maximization problem
            action_df = knobDF2action(action_df)  # normalize
        else:
            raise AutotuneError('no initial data provided for GP')
        record_num = len(tpsL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.float64)
        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))
        for global_step in range(NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                internal_metrics, initial_metrics, resource = self.env.initialize(collect_cpu_remote)
                logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(
                    resource, initial_metrics[0], initial_metrics[1], initial_metrics[2]
                ))

                latest_db_size = self.env.get_db_size()
                if latest_db_size > 4 * db_size:  # 4 times larger than it's original size
                    logger.warning('[Database Size Warning]. Your database {} size now is {}. We recommend you to restart your training task!'.format(
                        self.env.dbname, latest_db_size))
                else:
                    logger.info('[Database Size Warning]. Your database {} size now is {}. You are all good.'.format(
                        self.env.dbname, latest_db_size))

            if self.workload_map:
                matched_workload = self.map_workload(
                    X_scaled, internalm_matrix)
                matched_action_df, matched_tps = get_action_data_from_res(
                    '{}/{}'.format(self.output_log, matched_workload))
                matched_action_df = knobDF2action(
                    matched_action_df)  # normalize

                record_num = len(matched_tps)
                matched_X_scaled = matched_action_df[:record_num, :]

                normalized = StandardScaler()
                matched_tps = matched_tps + tps
                matched_y_scaled = normalized.fit_transform(
                    np.array(matched_tps).reshape(-1, 1))
                matched_X_scaled = np.vstack((matched_X_scaled, X_scaled))
                action, ypreds, sigmas = get_action_gp(matched_X_scaled, matched_y_scaled)
                logger.info('[GP-BOTORCH] Action: {}'.format(action))
                train_x = torch.tensor(matched_X_scaled)
                train_obj = torch.tensor(matched_y_scaled)
            else:
                tps_real = np.array(tpsL).reshape(-1, 1)
                cpu_real = np.array(cpuL).reshape(-1, 1)
                Y = np.hstack((tps_real, cpu_real))
                scaler = StandardScaler()
                Y_scaled = scaler.fit_transform(Y)
                train_x = torch.tensor(X_scaled)
                train_y = torch.tensor(Y_scaled)
                constraint_tmp = np.array([[self.tps_constrained, 0],])
                scaled_constraint = scaler.transform(constraint_tmp)[0][0]
            model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            constraints = {0: (scaled_constraint, None)}
            feas_y = train_y[torch.where(train_y[:, 0] >= scaled_constraint)]
            best_f = feas_y[:, 1].min() # suppose feasible points exist
            inverse_tmp = scaler.inverse_transform(np.array([[0, best_f],]))
            logger.info("[DEBUG] best_f: {}".format(inverse_tmp[0][1]))
            cei = ConstrainedExpectedImprovement(
                model,
                best_f=best_f,
                objective_index=1,
                constraints=constraints,
                maximize=False
            )
            candidates, acq_value = optimize_acqf(
                cei,
                bounds=torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], device=device, dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
            new_x = candidates.detach()
            self.CEI_debug(cei, candidates, global_step)
            action = new_x.squeeze(0).numpy()
            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))
            metrics, internal_metrics, resource = self.env.step(current_knob, global_step, collect_cpu_remote)
            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric cpu:{} tps:{} lat:{} qps:{}]'.format(
                global_step, resource, metrics[0], metrics[1], metrics[2]))
            tpsL.append(metrics[0])
            cpuL.append(resource)
            X_scaled = np.vstack((X_scaled, new_x.numpy()))
            logger.info('[GP-BOTORCH] Action: {}'.format(action))

            if internalm_matrix is not None:
                internal_metrics_tmp = convert_65IM_to_51IM(internal_metrics)
                internalm_matrix = np.vstack((internalm_matrix, internal_metrics_tmp.reshape(1, internalm_matrix.shape[1])))
            res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(current_knob), str(metrics[0]), str(metrics[1]), str(metrics[2]),
                resource, '', '', '', '', list(internal_metrics))
            '''if self.tps_constrained < metrics[0]:
                self.tps_constrained = metrics[0]
                logger.info("condtraint is changed to {}".format(metrics[0]))'''
            logger.info("{} is recorded in {}".format(res, fn))
            f.write(res)

            # TODO with enough data, no workload mapping
            if len(tpsL) >= 50:
                self.workload_map = False

            time.sleep(30)

        f.close()
        return

    def tune_WGPE_multiOutput(self, collect_cpu_remote):
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        default_knobs = self.env.default_knobs
        default_action = knob2action(default_knobs).reshape(1, -1)
        default_tpsL = []
        f = open(fn, 'a')
        internal_metrics, initial_metrics, resource = self.env.initialize(collect_cpu_remote)
        logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(resource,
            initial_metrics[0], initial_metrics[1], initial_metrics[2]))
        if self.default_constraint:
            self.tps_constrained = initial_metrics[0] * 0.95
            logger.info('[GP-BOTORCH][Episode: 1][Step: 0 tps_constraint: {}]'.format(self.tps_constrained))

        res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(self.env.default_knobs), str(initial_metrics[0]), str(initial_metrics[1]),
                                        str(initial_metrics[2]),resource, '', '', '', '', list(internal_metrics))
        logger.info("{} is recorded in {}".format(res, fn))
        f.write(res)
        f.close()
        default_tpsL.append(initial_metrics[0])
        dis2constraint = np.abs(initial_metrics[0] - self.tps_constrained)
        benchmark_action = default_action
        benchmark_tps = initial_metrics[0]
        f = open(fn, 'a')
        if self.rl_log != '':
            action_df, tps = get_action_data(self.rl_log)
        elif self.lhs_log != '':
            if self.workload_map:
                action_df, tps, internalm_matrix = get_data_for_mapping(
                    self.lhs_log)
            else:
                action_df, tpsL, cpuL, _ = get_action_data_from_res_cpu(self.lhs_log)
                # cpuL = [-cpu for cpu in cpuL] #botorch assumes a maximization problem
            action_df = knobDF2action(action_df)  # normalize
        else:
            raise AutotuneError('no initial data provided for GP')
        record_num = len(tpsL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.float64)
        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))

        for global_step in range(NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                internal_metrics, initial_metrics, resource = self.env.initialize(True)
                logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(
                    resource, initial_metrics[0], initial_metrics[1], initial_metrics[2]
                ))

                latest_db_size = self.env.get_db_size()
                if latest_db_size > 4 * db_size:  # 4 times larger than it's original size
                    logger.warning('[Database Size Warning]. Your database {} size now is {}. We recommend you to restart your training task!'.format(
                        self.env.dbname, latest_db_size))
                else:
                    logger.info('[Database Size Warning]. Your database {} size now is {}. You are all good.'.format(
                        self.env.dbname, latest_db_size))

            tps_real = np.array(tpsL).reshape(-1, 1)
            cpu_real = np.array(cpuL).reshape(-1, 1)
            Y = np.hstack((tps_real, cpu_real))
            scaler_tps = StandardScaler()
            scaler_cpu = StandardScaler()
            scaler_tps.fit(Y[:, 0].reshape(-1, 1))
            # Y[:, 0] = scaler_tps.transform(Y[: ,0].reshape(-1, 1)).flatten()
            scaler_cpu.fit(Y[:, 1].reshape(-1, 1))
            train_x = torch.tensor(X_scaled)
            train_y = torch.tensor(Y)
            model, return_constraint, constraints_all = initialize_WGPE_model_multiOutput(train_x, train_y.reshape(-1, 2),
                                        self.gp_tps_model_dir, self.gp_cpu_model_dir, self.tps_constrained)
            tps_mean = sum(tpsL) / len(tpsL)
            if not return_constraint:
                if self.tps_constrained / benchmark_tps < 0.99:
                    scaler =  self.tps_constrained / benchmark_tps
                elif self.tps_constrained > tps_mean and benchmark_tps > tps_mean:
                    scaler = (self.tps_constrained - tps_mean) / (benchmark_tps - tps_mean)
                else:
                    scaler = self.tps_constrained / benchmark_tps
                scaled_constraint = model.posterior(torch.tensor(benchmark_action).double()).mean[0][
                                            0].item() * scaler
            else:
                scaled_constraint = constraints_all
            constraints = {0: (scaled_constraint, None)}
            logger.info("scaled_constraint: {}".format(scaled_constraint))
            feas_y = Y[np.where(Y[:, 0] > self.tps_constrained)]
            best_f = feas_y[:, 1].min().item()
            best_f_scaled = scaler_cpu.transform(np.array(best_f).reshape(1, -1)).flatten()[0]  # suppose feasible points exist
            logger.info("[DEBUG] best_f: {}, best_f_scaled: {}".format(best_f, best_f_scaled))
            cei = ConstrainedExpectedImprovement(
                model,
                best_f=best_f_scaled,
                objective_index=1,
                constraints=constraints,
                maximize=False
            )

            candidates, acq_value = optimize_acqf(
                cei,
                bounds=torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], device=device, dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
            new_x = candidates.detach()
            self.CEI_debug(cei, candidates, global_step)
            action = new_x.squeeze(0).numpy()
            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))
            metrics, internal_metrics, resource = self.env.step(current_knob, global_step, collect_cpu_remote)
            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric cpu:{} tps:{} lat:{} qps:{}]'.format(
                global_step, resource, metrics[0], metrics[1], metrics[2]))
            tpsL.append(metrics[0])
            cpuL.append(resource)
            X_scaled = np.vstack((X_scaled, new_x.numpy()))
            logger.info('[GP-BOTORCH] Action: {}'.format(action))
            res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(current_knob), str(metrics[0]), str(metrics[1]), str(metrics[2]),
                resource, '', '', '', '', list(internal_metrics))
            logger.info("{} is recorded in {}".format(res, fn))
            f.write(res)

            # TODO with enough data, no workload mapping
            if len(tpsL) >= 50:
                self.workload_map = False
            if np.abs(metrics[0] - self.tps_constrained) < dis2constraint:
                dis2constraint = np.abs(metrics[0] - self.tps_constrained)
                benchmark_action = action.reshape(1, -1)
                benchmark_tps = metrics[0]

            time.sleep(30)

        f.close()
        return

    def tune_WGPE2(self):  # use constraint_scaled relative to default
        default_knobs = self.env.default_knobs
        default_action = knob2action(default_knobs).reshape(1, -1)
        default_tpsL = []
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        f = open(fn, 'a')
        internal_metrics, initial_metrics, resource = self.env.initialize(True)
        if self.default_constraint:
            self.tps_constrained = initial_metrics[0] * 0.95
            logger.info('[GP-BOTORCH][Episode: 1][Step: 0 tps_constraint: {}]'.format(self.tps_constrained))

        logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(resource,
                                                                                      initial_metrics[0],
                                                                                      initial_metrics[1],
                                                                                      initial_metrics[2]))
        res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(self.env.default_knobs),
                                                           str(initial_metrics[0]), str(initial_metrics[1]),
                                                           str(initial_metrics[2]), resource, '', '', '', '',
                                                           list(internal_metrics))

        logger.info("{} is recorded in {}".format(res, fn))
        default_tpsL.append(initial_metrics[0])
        dis2constraint = np.abs(initial_metrics[0] - self.tps_constrained)
        benchmark_action = default_action
        benchmark_tps = initial_metrics[0]
        f.write(res)
        f.close()
        f = open(fn, 'a')
        if self.rl_log != '':
            action_df, tps = get_action_data(self.rl_log)
        elif self.lhs_log != '':
            if self.workload_map:
                action_df, tps, internalm_matrix = get_data_for_mapping(
                    self.lhs_log)
            else:
                action_df, tpsL, cpuL, _ = get_action_data_from_res_cpu(self.lhs_log)
                # cpuL = [-cpu for cpu in cpuL] #botorch assumes a maximization problem
            action_df = knobDF2action(action_df)  # normalize
        else:
            raise AutotuneError('no initial data provided for GP')
        record_num = len(tpsL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.float64)
        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))
        for global_step in range(NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                internal_metrics, initial_metrics, resource = self.env.initialize(True)
                logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(
                    resource, initial_metrics[0], initial_metrics[1], initial_metrics[2]
                ))

                latest_db_size = self.env.get_db_size()
                if latest_db_size > 4 * db_size:  # 4 times larger than it's original size
                    logger.warning(
                        '[Database Size Warning]. Your database {} size now is {}. We recommend you to restart your training task!'.format(
                            self.env.dbname, latest_db_size))
                else:
                    logger.info(
                        '[Database Size Warning]. Your database {} size now is {}. You are all good.'.format(
                            self.env.dbname, latest_db_size))

            tps_real = np.array(tpsL).reshape(-1, 1)
            cpu_real = np.array(cpuL).reshape(-1, 1)
            Y = np.hstack((tps_real, cpu_real))
            scaler_tps = StandardScaler()
            scaler_cpu = StandardScaler()
            scaler_tps.fit(Y[:10, 0].reshape(-1, 1))
            # Y[:, 0] = scaler_tps.transform(Y[: ,0].reshape(-1, 1)).flatten()
            scaler_cpu.fit(Y[:, 1].reshape(-1, 1))
            train_x = torch.tensor(X_scaled)
            train_y = torch.tensor(Y)
            # scaled_constraint = scaler_tps.transform(np.array(self.tps_constrained).reshape(-1,1)).flatten()[0]
            # scaled_constraint = (self.tps_constrained-11395.0)/5847.8
            model_tps = initialize_WGPE_model(train_x, train_y[:, 0].reshape(-1, 1), self.gp_tps_model_dir)
            model_cpu = initialize_WGPE_model(train_x, train_y[:, 1].reshape(-1, 1), self.gp_cpu_model_dir)
            default_tps = sum(default_tpsL) / len(default_tpsL)
            scaled_constraint = model_tps(torch.tensor(benchmark_action).double()).mean.item() * (
                    self.tps_constrained / benchmark_tps)
            # model = SingleTaskGP(train_x, train_y)
            model = ModelListGP(model_tps, model_cpu)
            constraints = {0: (scaled_constraint, None)}
            logger.info("scaled_constraint: {}".format(scaled_constraint))
            feas_y = train_y[torch.where(train_y[:, 0] > self.tps_constrained)]
            best_f = feas_y[:, 1].min().item()
            best_f_scaled = scaler_cpu.transform(np.array(best_f).reshape(1, -1)).flatten()[
                0]  # suppose feasible points exist
            # inverse_tmp = scaler.inverse_transform( best_f)
            logger.info("[DEBUG] best_f: {}, best_f_scaled: {}".format(best_f, best_f_scaled))
            cei = ConstrainedExpectedImprovement(
                model,
                best_f=best_f_scaled,
                objective_index=1,
                constraints=constraints,
                maximize=False
            )
            candidates, acq_value = optimize_acqf(
                cei,
                bounds=torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], device=device,
                                    dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
            new_x = candidates.detach()
            action = new_x.squeeze(0).numpy()
            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))
            self.cei_debug(cei, candidates, 0)
            metrics, internal_metrics, resource = self.env.step(current_knob, global_step, True)
            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric cpu:{} tps:{} lat:{} qps:{}]'.format(
                global_step, resource, metrics[0], metrics[1], metrics[2]))
            tpsL.append(metrics[0])
            cpuL.append(resource)
            X_scaled = np.vstack((X_scaled, new_x.numpy()))
            logger.info('[GP-BOTORCH] Action: {}'.format(action))
            res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(current_knob),
                                                               str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                                               resource, '', '', '', '', list(internal_metrics))
            logger.info("{} is recorded in {}".format(res, fn))
            default_tpsL.append(metrics[0])
            if np.abs(metrics[0] - self.tps_constrained) < dis2constraint:
                dis2constraint = np.abs(metrics[0] - self.tps_constrained)
                benchmark_action = action.reshape(1, -1)
                benchmark_tps = metrics[0]

            f.write(res)
            time.sleep(30)

        f.close()
        return

    def combine_workload(self, y_target, y_workload):
        y_target = np.array(y_target).reshape(-1, 1)
        y_workload = np.array(y_workload).reshape(-1, 1)
        if y_target.shape[0] < 5:  # FIXME
            # FIXME : if there are fewer than 5 target results so far
            # then scale the y values (metrics) using the workload's
            # y_scaler. I'm not sure if 5 is the right cutoff.
            y_target_scaler = None
            y_workload_scaler = StandardScaler()
            y_matrix = np.vstack([y_workload, y_target])
            y_scaled = y_workload_scaler.fit_transform(y_matrix)
            scaler = y_workload_scaler
        else:
            # FIXME : otherwise try to compute a separate y_scaler for
            # the target and scale them separately.
            try:
                y_target_scaler = StandardScaler()
                y_workload_scaler = StandardScaler()
                y_target_scaled = y_target_scaler.fit_transform(y_target)
                y_workload_scaled = y_workload_scaler.fit_transform(y_workload)
                y_scaled = np.vstack([y_workload_scaled, y_target_scaled])
                scaler = y_target_scaler
            except ValueError:
                y_target_scaler = None
                y_workload_scaler = StandardScaler()
                y_scaled = y_workload_scaler.fit_transform(y_target)
                scaler = y_workload_scaler
        return y_scaled, scaler

    def tune_Ottertune(self, collect_cpu_remote):
        self.workload_map = True
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        f = open(fn, 'a')
        internal_metrics, initial_metrics, resource = self.env.initialize(collect_cpu_remote)
        logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(resource,
            initial_metrics[0], initial_metrics[1], initial_metrics[2]))
        if self.default_constraint:
            self.tps_constrained = initial_metrics[0]*0.95
            logger.info('[GP-BOTORCH][Episode: 1][Step: 0 tps_constraint: {}]'.format(self.tps_constrained))
        res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(self.env.default_knobs), str(initial_metrics[0]), str(initial_metrics[1]),
                                        str(initial_metrics[2]),resource, '', '', '', '', list(internal_metrics))

        logger.info("{} is recorded in {}".format(res, fn))
        f.write(res)
        f.close()
        f = open(fn, 'a')
        internalm_matrix = None

        action_df, tpsL, cpuL, internalm_matrix = get_action_data_from_res_cpu(self.lhs_log)
        action_df = knobDF2action(action_df)  # normalize

        record_num = len(tpsL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.float64)
        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))
        for global_step in range(NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                internal_metrics, initial_metrics, resource = self.env.initialize(collect_cpu_remote)
                logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(
                    resource, initial_metrics[0], initial_metrics[1], initial_metrics[2]
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
                matched_action_df, matched_tps, matched_cpu, _ = get_action_data_from_res_cpu(
                    '{}/{}'.format(self.output_log, matched_workload))
                matched_action_df = knobDF2action(
                    matched_action_df)  # normalize

                record_num = len(matched_tps)
                matched_X_scaled = matched_action_df[:record_num, :]
                matched_X_scaled = np.vstack((matched_X_scaled, X_scaled))
                tps_all, scaler_tps = self.combine_workload(tpsL, matched_tps)
                cpu_all, scaler_cpu = self.combine_workload(cpuL, matched_cpu)
                tps_real = np.array(tps_all).reshape(-1, 1)
                cpu_real = np.array(cpu_all).reshape(-1, 1)
            else:
                matched_cpu = []
                scaler_tps, scaler_cpu = StandardScaler(), StandardScaler()
                tps_real = np.array(tpsL).reshape(-1, 1)
                cpu_real = np.array(cpuL).reshape(-1, 1)
                tps_real = scaler_tps.fit_transform(tps_real)
                cpu_real = scaler_cpu.fit_transform(cpu_real)
                matched_X_scaled = X_scaled

            Y = np.hstack((tps_real, cpu_real))
            train_x = torch.tensor(matched_X_scaled)
            train_y = torch.tensor(Y)
            constraint_tmp = np.array(self.tps_constrained).reshape(-1, 1)
            scaled_constraint = scaler_tps.transform(constraint_tmp).flatten()[0]

            model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            constraints = {0: (scaled_constraint, None)}
            train_y_target = train_y[len(matched_cpu):, ]
            feas_y = train_y_target[torch.where(train_y_target[:,0 ] > scaled_constraint)][:,1 ]
            best_f = feas_y.min() # suppose feasible points exist
            inverse_tmp = scaler_cpu.inverse_transform(np.array(best_f).reshape(-1, 1)).flatten()[0]
            logger.info("[DEBUG] best_f: {}".format(inverse_tmp))
            cei = ConstrainedExpectedImprovement(
                model,
                best_f=best_f,
                objective_index=1,
                constraints=constraints,
                maximize=False
            )
            candidates, acq_value = optimize_acqf(
                cei,
                bounds=torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], device=device, dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
            new_x = candidates.detach()
            self.CEI_debug(cei, candidates, global_step)
            action = new_x.squeeze(0).numpy()
            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))
            metrics, internal_metrics, resource = self.env.step(current_knob, global_step, collect_cpu_remote)
            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric cpu:{} tps:{} lat:{} qps:{}]'.format(
                global_step, resource, metrics[0], metrics[1], metrics[2]))
            tpsL.append(metrics[0])
            cpuL.append(resource)
            X_scaled = np.vstack((X_scaled, new_x.numpy()))
            logger.info('[GP-BOTORCH] Action: {}'.format(action))

            internalm_matrix = np.vstack((internalm_matrix, internal_metrics.reshape(1, internalm_matrix.shape[1])))
            res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(current_knob), str(metrics[0]), str(metrics[1]), str(metrics[2]),
                resource, '', '', '', '', list(internal_metrics))
            logger.info("{} is recorded in {}".format(res, fn))
            f.write(res)

            # TODO with enough data, no workload mapping
            if len(tpsL) >= 50:
                self.workload_map = False

            time.sleep(30)

        f.close()
        return

    def tune_GP(self, collect_cpu_remote):
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        f = open(fn, 'a')
        internal_metrics, initial_metrics, resource = self.env.initialize(collect_cpu_remote)
        logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(resource,
            initial_metrics[0], initial_metrics[1], initial_metrics[2]))
        if self.default_constraint:
            self.tps_constrained = initial_metrics[0] * 0.95
            logger.info('[GP-BOTORCH][Episode: 1][Step: 0 tps_constraint: {}]'.format(self.tps_constrained))

        res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(self.env.default_knobs), str(initial_metrics[0]), str(initial_metrics[1]),
                                        str(initial_metrics[2]),resource, '', '', '', '', list(internal_metrics))

        logger.info("{} is recorded in {}".format(res, fn))
        f.write(res)
        f.close()
        f = open(fn, 'a')
        internalm_matrix = None
        if self.rl_log != '':
            action_df, tps = get_action_data(self.rl_log)
        elif self.lhs_log != '':
            if self.workload_map:
                action_df, tps, internalm_matrix = get_data_for_mapping(
                    self.lhs_log)
            else:
                action_df, tpsL, cpuL, _ = get_action_data_from_res_cpu(self.lhs_log)
                # cpuL = [-cpu for cpu in cpuL] #botorch assumes a maximization problem
            action_df = knobDF2action(action_df)  # normalize
        else:
            raise AutotuneError('no initial data provided for GP')
        record_num = len(tpsL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.float64)
        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))
        for global_step in range(NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                internal_metrics, initial_metrics, resource = self.env.initialize(collect_cpu_remote)
                logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(
                    resource, initial_metrics[0], initial_metrics[1], initial_metrics[2]
                ))
                latest_db_size = self.env.get_db_size()
                if latest_db_size > 4 * db_size:  # 4 times larger than it's original size
                    logger.warning('[Database Size Warning]. Your database {} size now is {}. We recommend you to restart your training task!'.format(
                        self.env.dbname, latest_db_size))
                else:
                    logger.info('[Database Size Warning]. Your database {} size now is {}. You are all good.'.format(
                        self.env.dbname, latest_db_size))

            tps_real = np.array(tpsL).reshape(-1, 1)
            cpu_real = np.array(cpuL).reshape(-1, 1)
            Y = np.hstack((tps_real, cpu_real))
            scaler = StandardScaler()
            Y_scaled = scaler.fit_transform(tps_real)
            train_x = torch.tensor(X_scaled)
            train_y = torch.tensor(Y_scaled)
            model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            feas_y = Y[np.where(Y[:, 0] > self.tps_constrained)]
            best_f = feas_y[:, 1].min() # suppose feasible points exist
            cpu_min = Y[:, 1].min()
            cpu_min_scaled = scaler.transform(np.array(cpu_min).reshape(-1, 1)).flatten()[0]
            logger.info("[DEBUG] best_f: {}".format(best_f))
            ei = ExpectedImprovement(
                model=model,
                best_f=cpu_min_scaled,
                maximize=False
            )
            candidates, acq_value = optimize_acqf(
                ei,
                bounds=torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], device=device, dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
            new_x = candidates.detach()
            action = new_x.squeeze(0).numpy()
            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))
            metrics, internal_metrics, resource = self.env.step(current_knob, global_step, collect_cpu_remote)
            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric cpu:{} tps:{} lat:{} qps:{}]'.format(
                global_step, resource, metrics[0], metrics[1], metrics[2]))
            tpsL.append(metrics[0])
            cpuL.append(resource)
            X_scaled = np.vstack((X_scaled, new_x.numpy()))
            logger.info('[GP-BOTORCH] Action: {}'.format(action))

            if internalm_matrix is not None:
                internal_metrics_tmp = convert_65IM_to_51IM(internal_metrics)
                internalm_matrix = np.vstack((internalm_matrix, internal_metrics_tmp.reshape(1, internalm_matrix.shape[1])))
            res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(current_knob), str(metrics[0]), str(metrics[1]), str(metrics[2]),
                resource, '', '', '', '', list(internal_metrics))
            logger.info("{} is recorded in {}".format(res, fn))
            f.write(res)

            # TODO with enough data, no workload mapping
            if len(tpsL) >= 50:
                self.workload_map = False

            time.sleep(30)

        f.close()
        return

    def tune_WGPE_meta(self, collect_cpu_remote):
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        default_knobs = self.env.default_knobs
        default_action = knob2action(default_knobs).reshape(1, -1)
        default_tpsL = []
        f = open(fn, 'a')
        internal_metrics, initial_metrics, resource = self.env.initialize(collect_cpu_remote)
        logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(resource,
            initial_metrics[0], initial_metrics[1], initial_metrics[2]))
        if self.default_constraint:
            self.tps_constrained = initial_metrics[0] * 0.95
            logger.info('[GP-BOTORCH][Episode: 1][Step: 0 tps_constraint: {}]'.format(self.tps_constrained))

        res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(self.env.default_knobs), str(initial_metrics[0]), str(initial_metrics[1]),
                                        str(initial_metrics[2]),resource, '', '', '', '', list(internal_metrics))
        logger.info("{} is recorded in {}".format(res, fn))
        f.write(res)
        f.close()
        default_tpsL.append(initial_metrics[0])
        dis2constraint = np.abs(initial_metrics[0] - self.tps_constrained)
        benchmark_action = default_action
        benchmark_tps = initial_metrics[0]
        f = open(fn, 'a')
        if self.rl_log != '':
            action_df, tps = get_action_data(self.rl_log)
        elif self.lhs_log != '':
            if self.workload_map:
                action_df, tps, internalm_matrix = get_data_for_mapping(
                    self.lhs_log)
            else:
                action_df, tpsL, cpuL, _ = get_action_data_from_res_cpu(self.lhs_log)
                # cpuL = [-cpu for cpu in cpuL] #botorch assumes a maximization problem
            action_df = knobDF2action(action_df)  # normalize
        else:
            raise AutotuneError('no initial data provided for GP')
        record_num = len(tpsL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.float64)
        db_size = self.env.get_db_size()
        logger.info('Original database size is {}.'.format(db_size))

        for global_step in range(10):
            logger.info('entering episode 0 step {}'.format(global_step))

            if global_step > 0 and global_step % 16 == 0:
                internal_metrics, initial_metrics, resource = self.env.initialize(True)
                logger.info('[Env initialized][Metrics cpu:{} tps:{} lat: {} qps: {}]'.format(
                    resource, initial_metrics[0], initial_metrics[1], initial_metrics[2]
                ))

                latest_db_size = self.env.get_db_size()
                if latest_db_size > 4 * db_size:  # 4 times larger than it's original size
                    logger.warning('[Database Size Warning]. Your database {} size now is {}. We recommend you to restart your training task!'.format(
                        self.env.dbname, latest_db_size))
                else:
                    logger.info('[Database Size Warning]. Your database {} size now is {}. You are all good.'.format(
                        self.env.dbname, latest_db_size))

            tps_real = np.array(tpsL).reshape(-1, 1)
            cpu_real = np.array(cpuL).reshape(-1, 1)
            Y = np.hstack((tps_real, cpu_real))
            scaler_tps = StandardScaler()
            scaler_cpu = StandardScaler()
            scaler_tps.fit(Y[:, 0].reshape(-1, 1))
            # Y[:, 0] = scaler_tps.transform(Y[: ,0].reshape(-1, 1)).flatten()
            scaler_cpu.fit(Y[:, 1].reshape(-1, 1))
            train_x = torch.tensor(X_scaled)
            train_y = torch.tensor(Y)
            if self.env.workload['name'] == 'workload_zoo':
                workload_type = self.env.workload_zoo_app.split('_')[0]
            else:
                workload_type = self.env.workload['name']
            model, return_constraint, constraints_all = initialize_WGPE_model_meta(train_x, train_y.reshape(-1, 2),
             workload_type ,self.gp_tps_model_dir, self.gp_cpu_model_dir)
            tps_mean = sum(tpsL) / len(tpsL)
            if not return_constraint:
                if self.tps_constrained / benchmark_tps < 0.99:
                    scaler =  self.tps_constrained / benchmark_tps
                elif self.tps_constrained > tps_mean and benchmark_tps > tps_mean:
                    scaler = (self.tps_constrained - tps_mean) / (benchmark_tps - tps_mean)
                else:
                    scaler = self.tps_constrained / benchmark_tps
                scaled_constraint = model.posterior(torch.tensor(benchmark_action).double()).mean[0][
                                            0].item() * scaler
            else:
                scaled_constraint = constraints_all
            constraints = {0: (scaled_constraint, None)}
            logger.info("scaled_constraint: {}".format(scaled_constraint))
            feas_y = Y[np.where(Y[:, 0] > self.tps_constrained)]
            best_f = feas_y[:, 1].min().item()
            best_f_scaled = scaler_cpu.transform(np.array(best_f).reshape(1, -1)).flatten()[0]  # suppose feasible points exist
            logger.info("[DEBUG] best_f: {}, best_f_scaled: {}".format(best_f, best_f_scaled))
            cei = ConstrainedExpectedImprovement(
                model,
                best_f=best_f_scaled,
                objective_index=1,
                constraints=constraints,
                maximize=False
            )

            candidates, acq_value = optimize_acqf(
                cei,
                bounds=torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], device=device, dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
            new_x = candidates.detach()
            self.CEI_debug(cei, candidates, global_step)
            action = new_x.squeeze(0).numpy()
            current_knob = generate_knobs(action, 'gp')
            logger.info('knobs generated: {}'.format(current_knob))
            metrics, internal_metrics, resource = self.env.step(current_knob, global_step, collect_cpu_remote)
            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric cpu:{} tps:{} lat:{} qps:{}]'.format(
                global_step, resource, metrics[0], metrics[1], metrics[2]))
            tpsL.append(metrics[0])
            cpuL.append(resource)
            X_scaled = np.vstack((X_scaled, new_x.numpy()))
            logger.info('[GP-BOTORCH] Action: {}'.format(action))
            res = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|65d\n'.format(stringify(current_knob), str(metrics[0]), str(metrics[1]), str(metrics[2]),
                resource, '', '', '', '', list(internal_metrics))
            logger.info("{} is recorded in {}".format(res, fn))
            f.write(res)

            # TODO with enough data, no workload mapping
            if len(tpsL) >= 50:
                self.workload_map = False
            if np.abs(metrics[0] - self.tps_constrained) < dis2constraint:
                dis2constraint = np.abs(metrics[0] - self.tps_constrained)
                benchmark_action = action.reshape(1, -1)
                benchmark_tps = metrics[0]

            time.sleep(30)

        f.close()
        return

    def tune(self, collect_cpu_remote=True):
        if self.method == 'GP_BOTORCH':
            return self.tune_GP_Botorch(collect_cpu_remote)
        elif self.method == 'WGPE':
            return self.tune_WGPE_multiOutput(collect_cpu_remote)

