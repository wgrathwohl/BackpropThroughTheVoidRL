import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape
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
    logprobs_zt = hard_samples_oh * scores2 + ((-1. * hard_samples_oh) + 1.) * scores3
    return hard_samples, tf.nn.softmax(logprobs_z / temp), tf.nn.softmax(logprobs_zt / temp)
    #return hard_samples, logprobs_z, logprobs_zt


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
