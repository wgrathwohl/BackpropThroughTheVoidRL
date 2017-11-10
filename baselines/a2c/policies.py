import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape
from baselines.acktr.utils import dense, kl_div
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import gym

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

def make_samples(logits, u, u2, log_temp, eps=1e-8):
    temp = tf.exp(log_temp)
    logprobs = tf.nn.log_softmax(logits)
    g = -tf.log(-tf.log(u + eps) + eps)
    scores = logprobs + g
    hard_samples = tf.argmax(scores,1)
    hard_samples_oh = tf.one_hot(hard_samples, scores.get_shape().as_list()[1])
    logprobs_z = scores

    g2 = -tf.log(-tf.log(u2 + eps) + eps)
    scores2 = logprobs + g2

    B = tf.reduce_sum(scores2 * hard_samples_oh, axis=1, keep_dims=True) - logprobs
    y = -1. * tf.log(u2) + tf.exp(-1. * B)
    g3 = -1. * tf.log(y)
    scores3 = g3 + logprobs
    # slightly biased…
    logprobs_zt = hard_samples_oh * scores2 + ((-1. * hard_samples_oh) + 1.) * scores3
    return hard_samples, tf.nn.softmax(logprobs_z / temp), tf.nn.softmax(logprobs_zt / temp)


class LinearPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        n_in, = ob_space.shape
        ob_shape = (None, n_in*nstack)
        nact = ac_space.n
        self.relaxed = False
        X = tf.placeholder(tf.float32, ob_shape) #obs
        U1 = tf.placeholder(tf.float32, (None, nact))
        U2 = tf.placeholder(tf.float32, (None, nact))
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=10, init_scale=np.sqrt(2))
            h2 = fc(h1, 'pi_fc2', nh=10, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact, act=lambda x:x)

            vh1 = fc(X, 'v_fc1', nh=10, init_scale=np.sqrt(2))
            vh2 = fc(vh1, 'v_fc2', nh=10, init_scale=np.sqrt(2))
            vf = fc(vh2, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0, s0, st0 = make_samples(pi, U1, U2, 0.0)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            s = (ob.shape[0], nact)
            u1 = np.random.random(s)
            u2 = np.random.random(s)
            a, v = sess.run([a0, v0], {X:ob, U1:u1, U2:u2})
            return a, u1, u2, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.a0 = a0
        self.U1 = U1
        self.U2 = U2


class RelaxedLinearPolicy(object):

    # noinspection PyPackageRequirements
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        n_in, = ob_space.shape
        ob_shape = (None, n_in*nstack)
        nact = ac_space.n
        self.relaxed = True
        X = tf.placeholder(tf.float32, ob_shape) #obs
        U1 = tf.placeholder(tf.float32, (None, nact))
        U2 = tf.placeholder(tf.float32, (None, nact))
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=10, init_scale=np.sqrt(2))
            h2 = fc(h1, 'pi_fc2', nh=10, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact, act=lambda x:x)
            log_temp = tf.get_variable("vf_log_temp", [], tf.float32, tf.constant_initializer(0.0))
            s = tf.get_variable("vf_scale", [], tf.float32, tf.constant_initializer(1.0))

        a0, s0, st0 = make_samples(pi, U1, U2, log_temp)
        self.initial_state = []  # not stateful

        with tf.variable_scope("model", reuse=reuse):
            vh1 = fc(tf.concat([X, s0], 1), 'vf_fc1', nh=10, init_scale=np.sqrt(2))
            vh2 = fc(vh1, 'vf_fc2', nh=10, init_scale=np.sqrt(2))
            vf = s * fc(vh2, 'vf', 1, act=lambda x:x)

        with tf.variable_scope("model", reuse=True):
            vh1_t = fc(tf.concat([X, st0], 1), 'vf_fc1', nh=10, init_scale=np.sqrt(2))
            vh2_t = fc(vh1_t, 'vf_fc2', nh=10, init_scale=np.sqrt(2))
            vf_t = s * fc(vh2_t, 'vf', 1, act=lambda x:x)

        v0 = vf_t[:, 0]

        def step(ob, *_args, **_kwargs):
            s = (ob.shape[0], nact)
            u1 = np.random.random(s)
            u2 = np.random.random(s)
            a, v = sess.run([a0, v0], {X:ob, U1:u1, U2:u2})
            return a, u1, u2, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.vf_t = vf_t
        self.step = step
        self.value = value
        self.a0 = a0
        self.U1 = U1
        self.U2 = U2
        
     
def pathlength(path):
    return path["reward"].shape[0]
      
      
class GaussianMlpPolicy(object):
    def __init__(self, sess, ob_dim, ac_dim, vf_lr=0.001, cv_lr=0.001, reuse=False):
        # Here we'll construct a bunch of expressions, which will be used in two places:
        # (1) When sampling actions
        # (2) When computing loss functions, for the policy update
        # Variables specific to (1) have the word "sampled" in them,
        # whereas variables specific to (2) have the word "old" in them
        self.relaxed = False
        self.X = tf.placeholder(tf.float32, shape=[None, ob_dim*2+ac_dim*2+2]) # batch of observations
        self.ob_no = tf.placeholder(tf.float32, shape=[None, ob_dim*2], name="ob") # batch of observations
        self.oldac_na = tf.placeholder(tf.float32, shape=[None, ac_dim], name="ac") # batch of actions previous actions
        oldac_dist = tf.placeholder(tf.float32, shape=[None, ac_dim*2], name="oldac_dist") # batch of actions previous action distributions
        
        with tf.variable_scope("model", reuse=reuse):
            h1 = tf.nn.tanh(dense(self.ob_no, 64, "pi_h1", weight_init=U.normc_initializer(1.0), bias_init=0.0))
            h2 = tf.nn.tanh(dense(h1, 64, "pi_h2", weight_init=U.normc_initializer(1.0), bias_init=0.0))
            mean_na = dense(h2, ac_dim, "pi", weight_init=U.normc_initializer(0.1), bias_init=0.0) # Mean control output
            self.logstd_1a = logstd_1a = tf.get_variable("logstd", [ac_dim], tf.float32, tf.zeros_initializer()) # Variance on outputs
            logstd_1a = tf.expand_dims(logstd_1a, 0)
            self.std_1a = tf.exp(logstd_1a)
            self.std_na = tf.tile(self.std_1a, [tf.shape(mean_na)[0], 1])
            ac_dist = tf.concat([tf.reshape(mean_na, [-1, ac_dim]), tf.reshape(self.std_na, [-1, ac_dim])], 1)
            sampled_ac_na = tf.random_normal(tf.shape(ac_dist[:,ac_dim:])) * ac_dist[:,ac_dim:] + ac_dist[:,:ac_dim] # This is the sampled action we'll perform.
            logprobsampled_n = - U.sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * U.sum(tf.square(ac_dist[:,:ac_dim] - sampled_ac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of sampled action
            self.logprob_n = - U.sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * U.sum(tf.square(ac_dist[:,:ac_dim] - self.oldac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of previous actions under CURRENT policy (whereas oldlogprob_n is under OLD policy)
            kl = U.mean(kl_div(oldac_dist, ac_dist, ac_dim))
        

            vh1 = tf.nn.elu(dense(self.X, 64, "vf_h1", weight_init=U.normc_initializer(1.0), bias_init=0))
            vh2 = tf.nn.elu(dense(vh1, 64, "vf_h2", weight_init=U.normc_initializer(1.0), bias_init=0))
            vpred_n = dense(vh2, 1, "vf", weight_init=None, bias_init=0)
            v0 = vpred_n[:, 0]
            self.vf_optim = tf.train.AdamOptimizer(vf_lr)
        
        def act(ob):
            ac, dist, logp = sess.run([sampled_ac_na, ac_dist, logprobsampled_n], {self.ob_no: ob[None]})  # Generate a new action and its logprob
            return ac[0], dist[0], logp[0]
        def value(obs, x):
            return sess.run(v0, {self.X: x, self.ob_no:obs})
        def preproc(path):
            l = pathlength(path)
            al = np.arange(l).reshape(-1,1)/10.0
            act = path["action_dist"].astype('float32')
            X = np.concatenate([path['observation'], act, al, np.ones((l, 1))], axis=1)
            return X
        def predict(obs, path):
            return value(obs, preproc(path))
        def compute_kl(ob, dist):
            return sess.run(kl, {self.ob_no: ob, oldac_dist: dist})
            
        self.mean = mean_na
        self.vf = v0
        self.act = act
        self.value = value
        self.preproc = preproc
        self.predict = predict
        self.compute_kl = compute_kl
        self.a0 = sampled_ac_na
        

class RelaxedGaussianMlpPolicy(object):
    def __init__(self, sess, ob_dim, ac_dim, vf_lr=0.001, cv_lr=0.001, reuse=False):
        # Here we'll construct a bunch of expressions, which will be used in two places:
        # (1) When sampling actions
        # (2) When computing loss functions, for the policy update
        # Variables specific to (1) have the word "sampled" in them,
        # whereas variables specific to (2) have the word "old" in them
        self.relaxed = True
        self.X = tf.placeholder(tf.float32, shape=[None, ob_dim*2+ac_dim*2+2]) # batch of observations
        self.ob_no = tf.placeholder(tf.float32, shape=[None, ob_dim*2], name="ob") # batch of observations
        self.oldac_na = tf.placeholder(tf.float32, shape=[None, ac_dim], name="ac") # batch of actions previous actions
        oldac_dist = tf.placeholder(tf.float32, shape=[None, ac_dim*2], name="oldac_dist") # batch of actions previous action distributions
        
        with tf.variable_scope("model", reuse=reuse):
            h1 = tf.nn.tanh(dense(self.ob_no, 64, "pi_h1", weight_init=U.normc_initializer(1.0), bias_init=0.0))
            h2 = tf.nn.tanh(dense(h1, 64, "pi_h2", weight_init=U.normc_initializer(1.0), bias_init=0.0))
            mean_na = dense(h2, ac_dim, "pi", weight_init=U.normc_initializer(0.1), bias_init=0.0) # Mean control output
            self.logstd_1a = logstd_1a = tf.get_variable("logstd", [ac_dim], tf.float32, tf.zeros_initializer()) # Variance on outputs
            logstd_1a = tf.expand_dims(logstd_1a, 0)
            self.std_1a = tf.exp(logstd_1a)
            self.std_na = tf.tile(self.std_1a, [tf.shape(mean_na)[0], 1])
            ac_dist = tf.concat([tf.reshape(mean_na, [-1, ac_dim]), tf.reshape(self.std_na, [-1, ac_dim])], 1)
            sampled_ac_na = tf.random_normal(tf.shape(ac_dist[:,ac_dim:])) * ac_dist[:,ac_dim:] + ac_dist[:,:ac_dim] # This is the sampled action we'll perform.
            logprobsampled_n = - U.sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * U.sum(tf.square(ac_dist[:,:ac_dim] - sampled_ac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of sampled action
            self.logprob_n = - U.sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * U.sum(tf.square(ac_dist[:,:ac_dim] - self.oldac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of previous actions under CURRENT policy (whereas oldlogprob_n is under OLD policy)
            kl = U.mean(kl_div(oldac_dist, ac_dist, ac_dim))
        
            vf1 = tf.nn.elu(dense(self.X, 64, 'vf_fc1', weight_init=U.normc_initializer(1.0), bias_init=0))
            vf2 = tf.nn.elu(dense(vf1, 64, 'vf_fc2', weight_init=U.normc_initializer(1.0), bias_init=0))
            vf = dense(vf2, 1, "vf", weight_init=None, bias_init=0)

            s = tf.get_variable("cv_scale", [], tf.float32, tf.constant_initializer(1.0))  
            cv1 = tf.nn.elu(dense(tf.concat([self.X, sampled_ac_na],1), 64, 'cv_fc1', weight_init=U.normc_initializer(1.0), bias_init=0))
            cv2 = tf.nn.elu(dense(cv1, 64, 'cv_fc2', weight_init=U.normc_initializer(1.0), bias_init=0))
            cv = s * dense(cv2, 1, "cv", weight_init=None, bias_init=0)

        v0 = vf[:, 0]
        self.vf_optim = tf.train.AdamOptimizer(vf_lr)
        self.cv_optim = tf.train.AdamOptimizer(cv_lr)    
        
        def act(ob):
            ac, dist, logp = sess.run([sampled_ac_na, ac_dist, logprobsampled_n], {self.ob_no: ob[None]})  # Generate a new action and its logprob
            return ac[0], dist[0], logp[0]
        def value(obs, x):
            return sess.run(v0, {self.X:x, self.ob_no:obs})
        def preproc(path):
            l = pathlength(path)
            al = np.arange(l).reshape(-1,1)/10.0
            act = path["action_dist"].astype('float32')
            X = np.concatenate([path['observation'], act, al, np.ones((l, 1))], axis=1)
            return X
        def predict(obs, path):
            return value(obs, preproc(path))
        def compute_kl(ob, dist):
            return sess.run(kl, {self.ob_no: ob, oldac_dist: dist})
            
        
        self.mean = mean_na
        self.vf = vf
        self.cv = cv
        self.act = act
        self.value = value
        self.preproc = preproc
        self.predict = predict
        self.compute_kl = compute_kl
        self.a0 = sampled_ac_na
