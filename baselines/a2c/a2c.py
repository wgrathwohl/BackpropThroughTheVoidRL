import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.a2c.utils import discount_with_dones, jacobian
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.policies import CnnPolicy
from baselines.a2c.utils import cat_entropy, mse
import random


def gs(x):
    return x.get_shape().as_list()


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear', logdir=None):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=train_model.a0)
        entropy = tf.reduce_sum(cat_entropy(train_model.pi))
        params = find_trainable_variables("model")
        tf.summary.histogram("vf", train_model.vf)
        tf.summary.histogram("R", R)

        if train_model.relaxed:
            pg_loss = tf.constant(0.0)
            oh_A = tf.one_hot(train_model.a0, ac_space.n)

            params = find_trainable_variables("model")
            policy_params = [v for v in params if "pi" in v.name]
            vf_params = [v for v in params if "vf" in v.name]
            entropy_grads = tf.gradients(entropy, policy_params)

            ddiff_loss = tf.reduce_sum(train_model.vf - train_model.vf_t)
            ddiff_grads = tf.gradients(ddiff_loss, policy_params)

            sm = tf.nn.softmax(train_model.pi)
            dlogp_dpi = oh_A * (1. - sm) + (1. - oh_A) * (-sm)
            pi_grads = -((tf.expand_dims(R, 1) - train_model.vf_t) * dlogp_dpi)
            pg_grads = tf.gradients(train_model.pi, policy_params, grad_ys=pi_grads)
            pg_grads = [pg - dg for pg, dg in zip(pg_grads, ddiff_grads)]

            pi_param_grads = tf.gradients(train_model.pi, policy_params, grad_ys=pi_grads)

            cv_grads = tf.concat([tf.reshape(p, [-1]) for p in pi_param_grads], 0)
            cv_grad_splits = tf.reduce_sum(tf.square(cv_grads))
            vf_loss = cv_grad_splits * vf_coef

            cv_grads = tf.gradients(vf_loss, vf_params)

            policy_grads = []
            for e_grad, p_grad, param in zip(entropy_grads, pg_grads, policy_params):
                grad = -e_grad * ent_coef + p_grad
                policy_grads.append(grad)
            grad_dict = {}

            for g, v in list(zip(policy_grads, policy_params))+list(zip(cv_grads, vf_params)):
                grad_dict[v] = g

            grads = [grad_dict[v] for v in params]
            print(grads)


        else:
            pg_loss = tf.reduce_sum((tf.stop_gradient(R) - tf.stop_gradient(train_model.vf)) * neglogpac)
            policy_params = [v for v in params if "pi" in v.name]
            pg_grads = tf.gradients(pg_loss, policy_params)

            vf_loss = tf.reduce_sum(mse(tf.squeeze(train_model.vf), R))
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
            grads = tf.gradients(loss, params)

        grads = list(zip(grads, params))

        ema = tf.train.ExponentialMovingAverage(.99)
        all_policy_grads = tf.concat([tf.reshape(g, [-1]) for g in pg_grads], 0)
        all_policy_grads_sq = tf.square(all_policy_grads)
        apply_mean_op = ema.apply([all_policy_grads, all_policy_grads_sq])
        em_mean = ema.average(all_policy_grads)
        em_mean_sq = ema.average(all_policy_grads_sq)
        em_var = em_mean_sq - tf.square(em_mean)
        em_log_var = tf.log(em_var + 1e-20)
        mlgv = tf.reduce_mean(em_log_var)

        for g, v in grads:
            print(v.name, g)
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name+"_grad", g)

        self.sum_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(logdir)

        trainer = tf.train.AdamOptimizer(learning_rate=LR, beta2=.99999)
        with tf.control_dependencies([apply_mean_op]):
            _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self._step = 0
        def train(obs, states, rewards, masks, u1, u2, values, summary=False):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {
                train_model.X:obs, train_model.U1:u1, train_model.U2:u2,
                ADV:advs, R:rewards, LR:cur_lr
            }
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            if summary:
                sum_str, policy_loss, value_loss, policy_entropy, lv, _ = sess.run(
                    [self.sum_op, pg_loss, vf_loss, entropy, mlgv, _train],
                    td_map
                )
                self.writer.add_summary(sum_str, self._step)
            else:
                policy_loss, value_loss, policy_entropy, lv, _ = sess.run(
                    [pg_loss, vf_loss, entropy, mlgv, _train],
                    td_map
                )
            self._step += 1
            return policy_loss, value_loss, policy_entropy, lv

        def save(save_path):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.model = model
        self.n_in, = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = nenv
        self.batch_ob_shape = (nenv*nsteps, self.n_in*nstack)
        self.obs = np.zeros((nenv, self.n_in*nstack))
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.n_in, axis=1)
        self.obs[:, -self.n_in:] = obs[:, :self.n_in]


    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


class RolloutRunner(Runner):
    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
        super().__init__(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)
        self._num_rollouts = 0
        self._num_steps = 0
        self.rewards = []

    def run(self):
        # reset env
        self.obs = np.zeros(self.obs.shape)
        obs = self.env.reset()
        self.update_obs(obs)

        # run env until all threads finish
        episode_over = [-1 for i in range(self.nenv)]
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_u1, mb_u2 = [], [], [], [], [], [], []
        mb_states = self.states
        step = 0
        while not all([e >= 0 for e in episode_over]):
            actions, u1, u2, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            mb_u1.append(u1)
            mb_u2.append(u2)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
                    if episode_over[n] == -1:
                        episode_over[n] = step
            self.update_obs(obs)
            mb_rewards.append(rewards)
            step += 1

        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        mb_u1 = np.asarray(mb_u1, dtype=np.float32).swapaxes(1, 0)
        mb_u2 = np.asarray(mb_u2, dtype=np.float32).swapaxes(1, 0)
        # discount/bootstrap off value fn
        _obs, _rewards, _actions, _values, _masks, _u1, _u2 = [], [], [], [], [], [], []
        for n, (obs, rewards, actions, values, dones, masks, u1, u2) in enumerate(zip(mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_masks, mb_u1, mb_u2)):
            # pull out data
            rewards = rewards.tolist()
            self.rewards.append(sum(rewards))
            actions = actions.tolist()
            values = values.tolist()
            dones = dones.tolist()
            masks = masks.tolist()
            u1, u2 = u1.tolist(), u2.tolist()
            # get length of this episode
            episode_length = episode_over[n]+1
            # crop out only played experience
            obs = obs[:episode_length]
            rewards = rewards[:episode_length]
            actions = actions[:episode_length]
            values = values[:episode_length]
            dones = dones[:episode_length]
            u1 = u1[:episode_length]
            u2 = u2[:episode_length]
            assert dones[-1] == True
            masks = masks[:episode_length]
            # discount the rewards
            rewards = discount_with_dones(rewards, dones, self.gamma)
            _obs.extend(obs)
            _rewards.extend(rewards)
            _actions.extend(actions)
            _values.extend(values)
            _masks.extend(masks)
            _u1.extend(u1)
            _u2.extend(u2)
        self.rewards = self.rewards[-100:]
        # make numpy
        mb_obs = np.asarray(_obs)
        mb_rewards = np.asarray(_rewards)
        mb_actions = np.asarray(_actions)
        mb_values = np.asarray(_values)
        mb_masks = np.asarray(_masks)
        mb_u1 = np.asarray(_u1)
        mb_u2 = np.asarray(_u2)
        self._num_rollouts += 1
        self._num_steps += len(rewards) * 4 # FRAME STACK
        ave_r = np.mean(self.rewards)
        #print("Episode {}, Ave R {}".format(self._num_rollouts, ave_r))
        logger.record_tabular("ave_r", ave_r)
        logger.record_tabular("last_r", self.rewards[-1])
        logger.record_tabular("num_rollouts", self._num_rollouts)
        logger.record_tabular("l", len(rewards) * 4)
        #logger.dump_tabular()
        END = False
        #print(self._num_steps, len(rewards))
        #if self._num_steps > 5000000:
        if np.mean(self.rewards) >= 195.:#195.:
            #if self._num_rollouts > 1000:
            logger.record_tabular("finished_in", self._num_rollouts)
            logger.record_tabular("total_steps", self._num_steps)
            logger.dump_tabular()
            END = True
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_u1, mb_u2, END


def learn(policy, env, seed, nsteps=5, nstack=1, total_timesteps=int(80e6),
        ent_coef=0.01, max_grad_norm=0.5,
          lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99,
          log_interval=100, logdir=None, bootstrap=False, args=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    lr = args.lr
    vf_coef = args.vf_coef

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, logdir=logdir)

    runner = RolloutRunner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        if True: #update % log_interval == 0 or update == 1:
            obs, states, rewards, masks, actions, values, u1, u2, END = runner.run()
            if END:
                break
            policy_loss, value_loss, policy_entropy, lv = model.train(obs, states, rewards, masks, u1, u2, values, summary=False)
            nseconds = time.time() - tstart
            fps = int((update * nbatch) / nseconds)

            ev = explained_variance(values, rewards)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("log_variance", lv)
            logger.dump_tabular()
        else:
            obs, states, rewards, masks, actions, values, u1, u2, END = runner.run()
            if END:
                break
            policy_loss, value_loss, policy_entropy, lv = model.train(obs, states, rewards, masks, u1, u2, values)
            nseconds = time.time() - tstart
            fps = int((update * nbatch) / nseconds)
    env.close()

if __name__ == '__main__':
    main()
