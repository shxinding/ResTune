#
# OtterTune - gp_tf.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Aug 18, 2016
LINK: https://github.com/cmu-db/ottertune/blob/master/server/analysis/gp_tf.py
'''
import queue
import gc
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import re
import ast
import pandas as pd
from pyDOE import lhs
from .knobs import logger

NUM_SAMPLES = 30

#  the number of selected tuning knobs
IMPORTANT_KNOB_NUMBER = 10

#  top K config with best performance put into prediction
TOP_NUM_CONFIG = 10

# ---CONSTRAINTS CONSTANTS---

#  Initial probability to flip categorical feature in apply_constraints
#  server/analysis/constraints.py
INIT_FLIP_PROB = 0.3

#  The probability that we flip the i_th categorical feature is
#  FLIP_PROB_DECAY * (probability we flip (i-1)_th categorical feature)
FLIP_PROB_DECAY = 0.5

# ---GPR CONSTANTS---
DEFAULT_LENGTH_SCALE = 1.0

DEFAULT_MAGNITUDE = 1.0

#  Max training size in GPR model
MAX_TRAIN_SIZE = 8000

#  Batch size in GPR model
BATCH_SIZE = 3000

#  Threads for TensorFlow config
NUM_THREADS = 4

# ---GRADIENT DESCENT CONSTANTS---
#  the maximum iterations of gradient descent
MAX_ITER = 500

DEFAULT_LEARNING_RATE = 0.01

# ---GRADIENT DESCENT FOR GPR---
#  a small bias when using training data points as starting points.
GPR_EPS = 1e-6

DEFAULT_RIDGE = 0.01

DEFAULT_EPSILON = 1e-6

DEFAULT_SIGMA_MULTIPLIER = 3.0

DEFAULT_MU_MULTIPLIER = 1.0

# ---GRADIENT DESCENT FOR DNN---
DNN_TRAIN_ITER = 500

DNN_EXPLORE = False

DNN_EXPLORE_ITER = 500

# noise scale for paramater space exploration
DNN_NOISE_SCALE_BEGIN = 0.1

DNN_NOISE_SCALE_END = 0.0

DNN_DEBUG = True

DNN_DEBUG_INTERVAL = 100

# ---DDPG CONSTRAINTS CONSTANTS---
#  Batch size in DDPG model
DDPG_BATCH_SIZE = 32

#  Learning rate of actor network
ACTOR_LEARNING_RATE = 0.001

#  Learning rate of critic network
CRITIC_LEARNING_RATE = 0.001

#  The impact of future reward on the decision
GAMMA = 0.1

#  The changing rate of the target network
TAU = 0.002

#LOG = get_analysis_logger(__name__)

class ParamConstraintHelper(object):

    def __init__(self, scaler, encoder=None, binary_vars=None,
                 init_flip_prob=0.3, flip_prob_decay=0.5):
        if 'inverse_transform' not in dir(scaler):
            raise Exception("Scaler object must provide function inverse_transform(X)")
        if 'transform' not in dir(scaler):
            raise Exception("Scaler object must provide function transform(X)")
        self.scaler_ = scaler
        if encoder is not None and len(encoder.n_values) > 0:
            self.is_dummy_encoded_ = True
            self.encoder_ = encoder.encoder
        else:
            self.is_dummy_encoded_ = False
        self.binary_vars_ = binary_vars
        self.init_flip_prob_ = init_flip_prob
        self.flip_prob_decay_ = flip_prob_decay

class GPRResult(object):

    def __init__(self, ypreds=None, sigmas=None):
        self.ypreds = ypreds
        self.sigmas = sigmas


class GPRGDResult(GPRResult):

    def __init__(self, ypreds=None, sigmas=None,
                 minl=None, minl_conf=None):
        super(GPRGDResult, self).__init__(ypreds, sigmas)
        self.minl = minl
        self.minl_conf = minl_conf


class GPR(object):

    def __init__(self, length_scale=1.0, magnitude=1.0, max_train_size=7000,
                 batch_size=3000, num_threads=4, check_numerics=True, debug=False):
        assert np.isscalar(length_scale)
        assert np.isscalar(magnitude)
        assert length_scale > 0 and magnitude > 0
        self.length_scale = length_scale
        self.magnitude = magnitude
        self.max_train_size_ = max_train_size
        self.batch_size_ = batch_size
        self.num_threads_ = num_threads
        self.check_numerics = check_numerics
        self.debug = debug
        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
        self.K_inv = None
        self.graph = None
        self.vars = None
        self.ops = None

    def build_graph(self):
        self.vars = {}
        self.ops = {}
        self.graph = tf.Graph()
        with self.graph.as_default():
            mag_const = tf.constant(self.magnitude,
                                    dtype=np.float32,
                                    name='magnitude')
            ls_const = tf.constant(self.length_scale,
                                   dtype=np.float32,
                                   name='length_scale')

            # Nodes for distance computation
            v1 = tf.placeholder(tf.float32, name="v1")
            v2 = tf.placeholder(tf.float32, name="v2")
            dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2), 1), name='dist_op')
            if self.check_numerics:
                dist_op = tf.check_numerics(dist_op, "dist_op: ")

            self.vars['v1_h'] = v1
            self.vars['v2_h'] = v2
            self.ops['dist_op'] = dist_op

            # Nodes for kernel computation
            X_dists = tf.placeholder(tf.float32, name='X_dists')
            ridge_ph = tf.placeholder(tf.float32, name='ridge')
            K_op = mag_const * tf.exp(-X_dists / ls_const)
            if self.check_numerics:
                K_op = tf.check_numerics(K_op, "K_op: ")
            K_ridge_op = K_op + tf.diag(ridge_ph)
            if self.check_numerics:
                K_ridge_op = tf.check_numerics(K_ridge_op, "K_ridge_op: ")

            self.vars['X_dists_h'] = X_dists
            self.vars['ridge_h'] = ridge_ph
            self.ops['K_op'] = K_op
            self.ops['K_ridge_op'] = K_ridge_op

            # Nodes for xy computation
            K = tf.placeholder(tf.float32, name='K')
            K_inv = tf.placeholder(tf.float32, name='K_inv')
            xy_ = tf.placeholder(tf.float32, name='xy_')
            yt_ = tf.placeholder(tf.float32, name='yt_')
            K_inv_op = tf.matrix_inverse(K)
            if self.check_numerics:
                K_inv_op = tf.check_numerics(K_inv_op, "K_inv: ")
            xy_op = tf.matmul(K_inv, yt_)
            if self.check_numerics:
                xy_op = tf.check_numerics(xy_op, "xy_: ")

            self.vars['K_h'] = K
            self.vars['K_inv_h'] = K_inv
            self.vars['xy_h'] = xy_
            self.vars['yt_h'] = yt_
            self.ops['K_inv_op'] = K_inv_op
            self.ops['xy_op'] = xy_op

            # Nodes for yhat/sigma computation
            K2 = tf.placeholder(tf.float32, name="K2")
            K3 = tf.placeholder(tf.float32, name="K3")
            yhat_ = tf.cast(tf.matmul(tf.transpose(K2), xy_), tf.float32)
            if self.check_numerics:
                yhat_ = tf.check_numerics(yhat_, "yhat_: ")
            sv1 = tf.matmul(tf.transpose(K2), tf.matmul(K_inv, K2))
            if self.check_numerics:
                sv1 = tf.check_numerics(sv1, "sv1: ")
            sig_val = tf.cast((tf.sqrt(tf.diag_part(K3 - sv1))), tf.float32)
            if self.check_numerics:
                sig_val = tf.check_numerics(sig_val, "sig_val: ")

            self.vars['K2_h'] = K2
            self.vars['K3_h'] = K3
            self.ops['yhat_op'] = yhat_
            self.ops['sig_op'] = sig_val

            # Compute y_best (min y)
            y_best_op = tf.cast(tf.reduce_min(yt_, 0, True), tf.float32)
            if self.check_numerics:
                y_best_op = tf.check_numerics(y_best_op, "y_best_op: ")
            self.ops['y_best_op'] = y_best_op

            sigma = tf.placeholder(tf.float32, name='sigma')
            yhat = tf.placeholder(tf.float32, name='yhat')

            self.vars['sigma_h'] = sigma
            self.vars['yhat_h'] = yhat

    def __repr__(self):
        rep = ""
        for k, v in sorted(self.__dict__.items()):
            rep += "{} = {}\n".format(k, v)
        return rep

    def __str__(self):
        return self.__repr__()

    def check_X_y(self, X, y):
        from sklearn.utils.validation import check_X_y

        if X.shape[0] > self.max_train_size_:
            raise Exception("X_train size cannot exceed {} ({})"
                            .format(self.max_train_size_, X.shape[0]))
        return check_X_y(X, y, multi_output=True,
                         allow_nd=True, y_numeric=True,
                         estimator="GPR")

    def check_fitted(self):
        if self.X_train is None or self.y_train is None \
                or self.xy_ is None or self.K is None:
            raise Exception("The model must be trained before making predictions!")

    @staticmethod
    def check_array(X):
        from sklearn.utils.validation import check_array
        return check_array(X, allow_nd=True, estimator="GPR")

    @staticmethod
    def check_output(X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))

    def fit(self, X_train, y_train, ridge=1.0):
        self._reset()
        X_train, y_train = self.check_X_y(X_train, y_train)
        self.X_train = np.float32(X_train)
        self.y_train = np.float32(y_train)
        sample_size = self.X_train.shape[0]

        if np.isscalar(ridge):
            ridge = np.ones(sample_size) * ridge
        assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1

        X_dists = np.zeros((sample_size, sample_size), dtype=np.float32)
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(
                            intra_op_parallelism_threads=self.num_threads_)) as sess:
            dist_op = self.ops['dist_op']
            v1, v2 = self.vars['v1_h'], self.vars['v2_h']
            for i in range(sample_size):
                X_dists[i] = sess.run(dist_op, feed_dict={v1: self.X_train[i], v2: self.X_train})

            K_ridge_op = self.ops['K_ridge_op']
            X_dists_ph = self.vars['X_dists_h']
            ridge_ph = self.vars['ridge_h']

            self.K = sess.run(K_ridge_op, feed_dict={X_dists_ph: X_dists, ridge_ph: ridge})

            K_ph = self.vars['K_h']

            K_inv_op = self.ops['K_inv_op']
            self.K_inv = sess.run(K_inv_op, feed_dict={K_ph: self.K})

            xy_op = self.ops['xy_op']
            K_inv_ph = self.vars['K_inv_h']
            yt_ph = self.vars['yt_h']
            self.xy_ = sess.run(xy_op, feed_dict={K_inv_ph: self.K_inv,
                                                  yt_ph: self.y_train})
        return self

    def predict(self, X_test):
        self.check_fitted()
        X_test = np.float32(GPR.check_array(X_test))
        test_size = X_test.shape[0]
        sample_size = self.X_train.shape[0]

        arr_offset = 0
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(
                            intra_op_parallelism_threads=self.num_threads_)) as sess:
            # Nodes for distance operation
            dist_op = self.ops['dist_op']
            v1 = self.vars['v1_h']
            v2 = self.vars['v2_h']

            # Nodes for kernel computation
            K_op = self.ops['K_op']
            X_dists = self.vars['X_dists_h']

            # Nodes to compute yhats/sigmas
            yhat_ = self.ops['yhat_op']
            K_inv_ph = self.vars['K_inv_h']
            K2 = self.vars['K2_h']
            K3 = self.vars['K3_h']
            xy_ph = self.vars['xy_h']

            while arr_offset < test_size:
                if arr_offset + self.batch_size_ > test_size:
                    end_offset = test_size
                else:
                    end_offset = arr_offset + self.batch_size_

                X_test_batch = X_test[arr_offset:end_offset]
                batch_len = end_offset - arr_offset

                dists1 = np.zeros([sample_size, batch_len])
                for i in range(sample_size):
                    dists1[i] = sess.run(dist_op, feed_dict={v1: self.X_train[i],
                                                             v2: X_test_batch})

                sig_val = self.ops['sig_op']
                K2_ = sess.run(K_op, feed_dict={X_dists: dists1})
                yhat = sess.run(yhat_, feed_dict={K2: K2_, xy_ph: self.xy_})
                dists2 = np.zeros([batch_len, batch_len])
                for i in range(batch_len):
                    dists2[i] = sess.run(dist_op, feed_dict={v1: X_test_batch[i], v2: X_test_batch})
                K3_ = sess.run(K_op, feed_dict={X_dists: dists2})

                sigma = np.zeros([1, batch_len], np.float32)
                sigma[0] = sess.run(sig_val, feed_dict={K_inv_ph: self.K_inv, K2: K2_, K3: K3_})
                sigma = np.transpose(sigma)
                yhats[arr_offset: end_offset] = yhat
                sigmas[arr_offset: end_offset] = sigma
                arr_offset = end_offset
        GPR.check_output(yhats)
        GPR.check_output(sigmas)
        return GPRResult(yhats, sigmas)

    def get_params(self, deep=True):
        return {"length_scale": self.length_scale,
                "magnitude": self.magnitude,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "xy_": self.xy_,
                "K": self.K,
                "K_inv": self.K_inv}

    def set_params(self, **parameters):
        for param, val in list(parameters.items()):
            setattr(self, param, val)
        return self

    def _reset(self):
        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
        self.K_inv = None
        self.graph = None
        self.build_graph()
        gc.collect()


class GPRGD(GPR):

    GP_BETA_UCB = "UCB"
    GP_BETA_CONST = "CONST"

    def __init__(self,
                 length_scale=1.0,
                 magnitude=1.0,
                 max_train_size=7000,
                 batch_size=3000,
                 num_threads=4,
                 learning_rate=0.001,
                 epsilon=1e-6,
                 max_iter=100,
                 sigma_multiplier=3.0,
                 mu_multiplier=1.0):
        super(GPRGD, self).__init__(length_scale=length_scale,
                                    magnitude=magnitude,
                                    max_train_size=max_train_size,
                                    batch_size=batch_size,
                                    num_threads=num_threads)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.sigma_multiplier = sigma_multiplier
        self.mu_multiplier = mu_multiplier
        self.X_min = None
        self.X_max = None
        #with tf.Session() as ss:
            #默认在当前py目录下的logs文件夹，没有会自己创建
            #wirter = tf.summary.FileWriter('logs/',ss.graph)
            #result = ss.run([mul,add])
            #print(result)

    def fit(self, X_train, y_train, X_min, X_max, ridge):  # pylint: disable=arguments-differ
        super(GPRGD, self).fit(X_train, y_train, ridge)
        self.X_min = X_min
        self.X_max = X_max

        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(
                            intra_op_parallelism_threads=self.num_threads_)) as sess:
            xt_ = tf.Variable(self.X_train[0], tf.float32)
            xt_ph = tf.placeholder(tf.float32)
            xt_assign_op = xt_.assign(xt_ph)
            init = tf.global_variables_initializer()
            sess.run(init)
            K2_mat = tf.transpose(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(
                tf.subtract(xt_, self.X_train), 2), 1) +  abs(GPR_EPS)), 0))
            if self.check_numerics is True:
                K2_mat = tf.check_numerics(K2_mat, "K2_mat: ")
            K2__ = tf.cast(self.magnitude * tf.exp(-K2_mat / self.length_scale), tf.float32)
            if self.check_numerics is True:
                K2__ = tf.check_numerics(K2__, "K2__: ")
            yhat_gd = tf.cast(tf.matmul(tf.transpose(K2__), self.xy_), tf.float32)
            if self.check_numerics is True:
                yhat_gd = tf.check_numerics(yhat_gd, message="yhat: ")
            sig_val = tf.cast((tf.sqrt(self.magnitude - tf.matmul(
                tf.transpose(K2__), tf.matmul(self.K_inv, K2__)))), tf.float32)
            if self.check_numerics is True:
                sig_val = tf.check_numerics(sig_val, message="sigma: ")
            #print("\nyhat_gd : %s", str(sess.run(yhat_gd)))
            #print("\nsig_val : %s", str(sess.run(sig_val)))

            loss = tf.squeeze(tf.subtract(self.mu_multiplier * yhat_gd,
                                          self.sigma_multiplier * sig_val))
            if self.check_numerics is True:
                loss = tf.check_numerics(loss, "loss: ")
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               epsilon=self.epsilon)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            train = optimizer.minimize(loss)

            self.vars['xt_'] = xt_
            self.vars['xt_ph'] = xt_ph
            self.ops['K2_mat'] = K2_mat
            self.ops['xt_assign_op'] = xt_assign_op
            self.ops['yhat_gd'] = yhat_gd
            self.ops['sig_val2'] = sig_val
            self.ops['loss_op'] = loss
            self.ops['train_op'] = train
            
        return self

    def predict(self, X_test, constraint_helper=None,  # pylint: disable=arguments-differ
                categorical_feature_method='hillclimbing',
                categorical_feature_steps=3):
        self.check_fitted()
        X_test = np.float32(GPR.check_array(X_test))
        test_size = X_test.shape[0]
        nfeats = self.X_train.shape[1]

        arr_offset = 0
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        minls = np.zeros([test_size, 1])
        minl_confs = np.zeros([test_size, nfeats])

        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(
                            intra_op_parallelism_threads=self.num_threads_)) as sess:
            while arr_offset < test_size:
                if arr_offset + self.batch_size_ > test_size:
                    end_offset = test_size
                else:
                    end_offset = arr_offset + self.batch_size_

                X_test_batch = X_test[arr_offset:end_offset]
                batch_len = end_offset - arr_offset

                xt_ = self.vars['xt_']
                #tf.summary.histogram('xt_',xt_)
                init = tf.global_variables_initializer()
                sess.run(init)

                #merged=tf.summary.merge_all()#合并所有的summary data的获取函数，merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
                #writer=tf.summary.FileWriter("logs2/",sess.graph)

                sig_val = self.ops['sig_val2']
                yhat_gd = self.ops['yhat_gd']
                loss = self.ops['loss_op']
                train = self.ops['train_op']

                xt_ph = self.vars['xt_ph']
                assign_op = self.ops['xt_assign_op']

                yhat = np.empty((batch_len, 1))
                sigma = np.empty((batch_len, 1))
                minl = np.empty((batch_len, 1))
                minl_conf = np.empty((batch_len, nfeats))
                for i in range(batch_len):
                    if self.debug is True:
                        print("-------------------------------------------")
                    yhats_it = np.empty((self.max_iter + 1,)) * np.nan
                    sigmas_it = np.empty((self.max_iter + 1,)) * np.nan
                    losses_it = np.empty((self.max_iter + 1,)) * np.nan
                    confs_it = np.empty((self.max_iter + 1, nfeats)) * np.nan

                    sess.run(assign_op, feed_dict={xt_ph: X_test_batch[i]})
                    step = 0
                    for step in range(self.max_iter):
                        if self.debug is True:
                            print("Batch %d, iter %d:", i, step)
                        yhats_it[step] = sess.run(yhat_gd)[0][0]
                        sigmas_it[step] = sess.run(sig_val)[0][0]
                        losses_it[step] = sess.run(loss)
                        confs_it[step] = sess.run(xt_)
                        if self.debug is True:
                            print("    yhat:  %s", str(yhats_it[step]))
                            print("    sigma: %s", str(sigmas_it[step]))
                            print("    loss:  %s", str(losses_it[step]))
                            print("    conf:  %s", str(confs_it[step]))

                        #sess.run(tf.gradients(loss, xt_))
                        sess.run(train)
                        #rs=sess.run(merged)#运行所有合并所有的图，获取summary data函数节点和graph是独立的，调用的时候也需要运行session
                        #writer.add_summary(rs,i)#把数据添加到文件中，每一次run的信息和得到的数据加到wr

                        # constraint Projected Gradient Descent
                        xt = sess.run(xt_)
                        xt_valid = np.minimum(xt, self.X_max)
                        xt_valid = np.maximum(xt_valid, self.X_min)

                        sess.run(assign_op, feed_dict={xt_ph: xt_valid})
                        if constraint_helper is not None:
                            xt_valid = constraint_helper.apply_constraints(sess.run(xt_))
                            sess.run(assign_op, feed_dict={xt_ph: xt_valid})
                            if categorical_feature_method == 'hillclimbing':
                                if step % categorical_feature_steps == 0:
                                    current_xt = sess.run(xt_)
                                    current_loss = sess.run(loss)
                                    new_xt = \
                                        constraint_helper.randomize_categorical_features(
                                            current_xt)
                                    sess.run(assign_op, feed_dict={xt_ph: new_xt})
                                    new_loss = sess.run(loss)
                                    if current_loss < new_loss:
                                        sess.run(assign_op, feed_dict={xt_ph: new_xt})
                            else:
                                raise Exception("Unknown categorial feature method: {}".format(
                                    categorical_feature_method))
                                    
                    if step == self.max_iter - 1:
                        # Record results from final iteration
                        yhats_it[-1] = sess.run(yhat_gd)[0][0]
                        sigmas_it[-1] = sess.run(sig_val)[0][0]
                        losses_it[-1] = sess.run(loss)
                        confs_it[-1] = sess.run(xt_)
                        assert np.all(np.isfinite(yhats_it))
                        assert np.all(np.isfinite(sigmas_it))
                        assert np.all(np.isfinite(losses_it))
                        assert np.all(np.isfinite(confs_it))

                    # Store info for conf with min loss from all iters
                    if np.all(~np.isfinite(losses_it)):
                        min_loss_idx = 0
                    else:
                        min_loss_idx = np.nanargmin(losses_it)
                    yhat[i] = yhats_it[min_loss_idx]
                    sigma[i] = sigmas_it[min_loss_idx]
                    minl[i] = losses_it[min_loss_idx]
                    minl_conf[i] = confs_it[min_loss_idx]

                minls[arr_offset:end_offset] = minl
                minl_confs[arr_offset:end_offset] = minl_conf
                yhats[arr_offset:end_offset] = yhat
                sigmas[arr_offset:end_offset] = sigma
                arr_offset = end_offset

        GPR.check_output(yhats)
        GPR.check_output(sigmas)
        GPR.check_output(minls)
        GPR.check_output(minl_confs)

        return GPRGDResult(yhats, sigmas, minls, minl_confs)
        

    @staticmethod
    def calculate_sigma_multiplier(t, ndim, bound=0.1):
        assert t > 0
        assert ndim > 0
        assert bound > 0 and bound <= 1
        beta = 2 * np.log(ndim * (t**2) * (np.pi**2) / 6 * bound)
        if beta > 0:
            beta = np.sqrt(beta)
        else:
            beta = 1
        return beta



def euclidean_mat(X, y, sess):
    x_n = X.shape[0]
    y_n = y.shape[0]
    z = np.zeros([x_n, y_n])
    for i in range(x_n):
        v1 = X[i]
        tmp = []
        for j in range(y_n):
            v2 = y[j]
            tmp.append(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2))))
        z[i] = (sess.run(tmp))
    return z


def gd_tf(xs, ys, xt, ridge, length_scale=1.0, magnitude=1.0, max_iter=50):
    LOG.debug("xs shape: %s", str(xs.shape))
    LOG.debug("ys shape: %s", str(ys.shape))
    LOG.debug("xt shape: %s", str(xt.shape))
    with tf.Graph().as_default():
        # y_best = tf.cast(tf.reduce_min(ys,0,True),tf.float32);   #array
        # yhat_gd = tf.check_numerics(yhat_gd, message="yhat: ")
        sample_size = xs.shape[0]
        nfeats = xs.shape[1]
        test_size = xt.shape[0]
        # arr_offset = 0
        ini_size = xt.shape[0]

        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        minl = np.zeros([test_size, 1])
        new_conf = np.zeros([test_size, nfeats])

        xs = np.float32(xs)
        ys = np.float32(ys)
        xt_ = tf.Variable(xt[0], tf.float32)

        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
        init = tf.global_variables_initializer()
        sess.run(init)

        ridge = np.float32(ridge)
        v1 = tf.placeholder(tf.float32, name="v1")
        v2 = tf.placeholder(tf.float32, name="v2")
        dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2), 1))

        tmp = np.zeros([sample_size, sample_size])
        for i in range(sample_size):
            tmp[i] = sess.run(dist, feed_dict={v1: xs[i], v2: xs})

        tmp = tf.cast(tmp, tf.float32)
        K = magnitude * tf.exp(-tmp / length_scale) + tf.diag(ridge)
        LOG.debug("K shape: %s", str(sess.run(K).shape))

        K2_mat = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(xt_, xs), 2), 1))
        K2_mat = tf.transpose(tf.expand_dims(K2_mat, 0))
        K2 = tf.cast(tf.exp(-K2_mat / length_scale), tf.float32)

        x = tf.matmul(tf.matrix_inverse(K), ys)
        x = sess.run(x)
        yhat_ = tf.cast(tf.matmul(tf.transpose(K2), x), tf.float32)
        sig_val = tf.cast((tf.sqrt(magnitude - tf.matmul(
            tf.transpose(K2), tf.matmul(tf.matrix_inverse(K), K2)))), tf.float32)

        #print('yhat shape: %s', str(sess.run(yhat_).shape))
        #print('sig_val shape: %s', str(sess.run(sig_val).shape))
        yhat_ = tf.check_numerics(yhat_, message='yhat: ')
        sig_val = tf.check_numerics(sig_val, message='sig_val: ')
        loss = tf.squeeze(tf.subtract(yhat_, sig_val))
        loss = tf.check_numerics(loss, message='loss: ')
        #optimizer = tf.train.GradientDescentOptimizer(0.1)
        print('loss: %s', str(sess.run(loss)))
        #optimizer = tf.train.AdamOptimizer(0.0000001)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               epsilon=self.epsilon)
        train = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(ini_size):
            assign_op = xt_.assign(xt[i])
            sess.run(assign_op)
            for step in range(max_iter):
                #print('sample #: %d, iter #: %d, loss: %s', i, step, str(sess.run(loss)))
                sess.run(train)
            yhats[i] = sess.run(yhat_)[0][0]
            sigmas[i] = sess.run(sig_val)[0][0]
            minl[i] = sess.run(loss)
            new_conf[i] = sess.run(xt_)
        return yhats, sigmas, minl, new_conf


def main():
    pass


def create_random_matrices(n_samples=3000, n_feats=12, n_test=4444):
    X_train = np.random.rand(n_samples, n_feats)
    y_train = np.random.rand(n_samples, 1)
    X_test = np.random.rand(n_test, n_feats)

    length_scale = np.random.rand()
    magnitude = np.random.rand()
    ridge = np.ones(n_samples) * np.random.rand()

    return X_train, y_train, X_test, length_scale, magnitude, ridge
def get_action_data(fn):
    # returns a DataFrame
    actions = []
    # [2019-09-01 07:35:48,984:tuner.py#L128:INFO]: [ddpg] Action: [0.516652   0.46629024 0.5140691  0.43496618 0.46725872 0.47590694
    #  0.47035056 0.510609   0.5435336  0.5607054  0.5314267  0.57516867
    #  0.48946124 0.42248294 0.48049828 0.5589055  0.5343374  0.59791833
    #  0.5194727  0.55978596 0.5200966  0.5583046  0.43084115 0.5462725
    #  0.54505014 0.4609037  0.5477881  0.4883114 ]
    flag = 0
    action = []
    with open(fn) as f:
        for line in f:
            if '[ddpg] Action:' in line and line[line.rfind('['):].find(']') != -1 and flag==0:
                s = line.rfind('[')
                list_str = line[s:].strip('\n')
                list_str = re.sub(r"\s{2,}", " ", list_str).replace(' ', ',')
                try:
                    action = ast.literal_eval(list_str)
                except:
                    print(line)
                actions.append(action)
            
            elif flag > 0 and line.find(']') != -1:
                line = line.strip()
                list_str = '[' + line
                list_str = re.sub(r"\s{2,}", " ", list_str).replace(' ', ',')
                flag = 0
                action.extend(ast.literal_eval(list_str))
                actions.append(action)
            elif flag > 0:
                line = line.strip()
                list_str = '[' + line.strip('\n') + ']'
                list_str = re.sub(r"\s{2,}", " ", list_str).replace(' ', ',')
                action.extend(ast.literal_eval(list_str))
                flag += 1
            elif '[ddpg] Action:' in line:
                flag = 1
                s = line.rfind('[')
                list_str = line[s:].strip('\n') + ']'
                #list_str = line[s:].strip('\n') 
                list_str = re.sub(r"\s{2,}", " ", list_str).replace(' ', ',')
                action = ast.literal_eval(list_str) 

    df = pd.DataFrame(actions)
    
    tps_list = []
    pat = r'ddpgEpisode: (\d+)Step: (\d+)Metric tps:(\d+.\d+) lat:(\d+.\d+) qps:(\d+.\d+)Reward: -*(\d+.\d+).*?'
    with open(fn) as f:
        for line in f:
            if 'Reward' in line:
                line = line.replace('[', '').replace(']', '')
                [(episode, step, tps, lat, qps, reward)] = re.findall(pat, line)
                tps_list.append(float(tps))

    return np.array(df),tps_list


def get_action_data_from_res(fn):
    f = open(fn)
    lines = f.readlines()
    f.close()
    metricsL = []
    tpsL = []
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
        metricsL.append(valueL)
        tpsL.append(tps)
    df = pd.DataFrame(metricsL, columns=nameL)

    return df, tpsL


def get_action_gp(X_scaled,y_scaled, sigma_multiplier_flexiable=DEFAULT_SIGMA_MULTIPLIER):
    sample_num=y_scaled.shape[0]
    
    X_min=np.zeros((X_scaled.shape[1],))
    X_max=np.ones((X_scaled.shape[1],))
    '''
    constraint_helper = ParamConstraintHelper(scaler=X_scaler,
                                              encoder=dummy_encoder,
                                              binary_vars=categorical_info['binary_vars'],
                                              init_flip_prob=INIT_FLIP_PROB,
                                              flip_prob_decay=FLIP_PROB_DECAY)'''
    model = GPRGD(length_scale=DEFAULT_LENGTH_SCALE,
                      magnitude=DEFAULT_MAGNITUDE,
                      max_train_size=MAX_TRAIN_SIZE,
                      batch_size=BATCH_SIZE,
                      num_threads=NUM_THREADS,
                      learning_rate=DEFAULT_LEARNING_RATE,
                      epsilon=DEFAULT_EPSILON,
                      max_iter=MAX_ITER,
                      sigma_multiplier=sigma_multiplier_flexiable,
                      mu_multiplier=DEFAULT_MU_MULTIPLIER)

    model.fit(X_scaled, y_scaled, X_min, X_max, ridge=DEFAULT_RIDGE)


    num_samples = NUM_SAMPLES
    X_samples = np.random.rand(num_samples, X_scaled.shape[1])
    q = queue.PriorityQueue()
    for x in range(0, y_scaled.shape[0]):
        q.put((y_scaled[x][0], x))
    i = 0
    while i < TOP_NUM_CONFIG:
        try:
            item = q.get_nowait()
            X_samples = np.vstack((X_samples, X_scaled[item[1]]))
            i = i + 1
        except queue.Empty:
            break


    #X_samples=X_scaled
    res = model.predict(X_samples)
    best_config_idx = np.argmin(res.minl.ravel())
    best_config = res.minl_conf[best_config_idx, :]
    return best_config, res.ypreds[best_config_idx], res.sigmas[best_config_idx]


def get_best_action_gp(X_scaled, y_scaled):
    lhs_num_samples = 1000  #it takes more than 10 minites for lhs to sample 1w ponits
    random_sample_num = 100000 #so we also use random to sample

    model = GPR(length_scale=DEFAULT_LENGTH_SCALE,
                  magnitude=DEFAULT_MAGNITUDE,
                  max_train_size=MAX_TRAIN_SIZE,
                  batch_size=BATCH_SIZE,
                  num_threads=NUM_THREADS)

    model.fit(X_scaled, y_scaled, ridge=0.5)

    X_samples_lhs = np.random.rand(random_sample_num , X_scaled.shape[1])

    X_samples_ran = lhs(X_scaled.shape[1], samples=lhs_num_samples, criterion='maximin')
    X_samples = np.vstack(( X_samples_lhs,  X_samples_ran))
    q = queue.PriorityQueue()
    for x in range(0, y_scaled.shape[0]):
        q.put((y_scaled[x][0], x))
    i = 0
    best_tps_in_queue = 0

    #add top number
    while i < TOP_NUM_CONFIG:
        try:
            item = q.get_nowait()
            if i==0:
                best_tps_in_queue = y_scaled[item[1]][0]
            X_samples = np.vstack((X_samples, X_scaled[item[1]]))
            i = i + 1
        except queue.Empty:
            break

    #inferance
    res = model.predict(X_samples)
    best_config_idx = np.argmin(res.ypreds.ravel())
    best_config = X_samples[best_config_idx]
    action, ypreds, sigmas = get_action_gp(X_scaled, y_scaled, 0)

    queue_generated = False
    log = ''
    if best_config_idx < lhs_num_samples:
        log = '[gp]: best action generated from lhs'
    elif best_config_idx < lhs_num_samples+random_sample_num:
        log = '[gp]: best action generated from random sampling'
    else:
        log = '[gp]: best action generated from queue'
        queue_generated = True

    if ypreds > res.ypreds[best_config_idx]:
        logger.info('[gp]: best action generated from gradient descent')
        queue_generated = False
        return action, ypreds, sigmas, queue_generated, best_tps_in_queue

    logger.info(log)
    return best_config, res.ypreds[best_config_idx], res.sigmas[best_config_idx], queue_generated, best_tps_in_queue

def get_pred_gp(X_scaled,y_scaled, X_samples, sigma_multiplier_flexiable=DEFAULT_SIGMA_MULTIPLIER):
    X_min = np.zeros((X_scaled.shape[1],))
    X_max = np.ones((X_scaled.shape[1],))

    model = GPR(length_scale=DEFAULT_LENGTH_SCALE,
                  magnitude=DEFAULT_MAGNITUDE,
                  max_train_size=MAX_TRAIN_SIZE,
                  batch_size=BATCH_SIZE,
                  num_threads=NUM_THREADS)

    model.fit(X_scaled, y_scaled, ridge=DEFAULT_RIDGE)

    res = model.predict(X_samples)
    return res.ypreds

