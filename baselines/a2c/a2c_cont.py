import numpy as np
import tensorflow as tf
import joblib
from baselines import logger
from baselines import common
from baselines.common import set_global_seeds
from baselines.acktr.filters import ZFilter
from baselines.a2c.utils import make_path, find_trainable_variables
from baselines.a2c.utils import mse


def gs(x):
    return x.get_shape().as_list()


class Model(object):

    def __init__(self, optim, policy, ob_dim, ac_dim, num_procs,
            max_grad_norm=0.5, lr=7e-4, vf_lr=0.001, cv_lr=0.001, cv_num=25,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear', logdir=None):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        A = tf.placeholder(tf.float32, [None, ac_dim], name="A")
        ADV = tf.placeholder(tf.float32, [None], name="ADV")
        R = tf.placeholder(tf.float32, [None], name="R")
        BS_R = tf.placeholder(tf.float32, [None], name="BS_R") # bootstrapped reward
    
        train_model = policy(sess, ob_dim, ac_dim, vf_lr, cv_lr, reuse=False)
        step_model = policy(sess, ob_dim, ac_dim, vf_lr, cv_lr, reuse=True)

        params = find_trainable_variables("model")
        tf.summary.histogram("vf", train_model.vf)
        
        pi_params = [v for v in params if "pi" in v.name]
        vf_params = [v for v in params if "vf" in v.name]
        
        logpac = train_model.logprob_n
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R)) 
        pg_loss = -tf.reduce_mean(ADV * logpac)
        tf.summary.scalar("vf_loss", vf_loss)

        if train_model.relaxed:
            ddiff_loss = tf.reduce_mean(train_model.cv)            
            ddiff_grads_mean = tf.gradients(ddiff_loss, train_model.mean)[0]
            ddiff_grads_std = tf.gradients(ddiff_loss, train_model.std_na)[0]
            
            dlogp_dmean = (A - train_model.mean)/ tf.square(train_model.std_na)
            dlogp_dstd = -1/train_model.std_na + 1/tf.pow(train_model.std_na,3)*tf.square(A-train_model.mean)            
            
            pi_grads_mean = -((tf.expand_dims(BS_R, 1) - train_model.vf - train_model.cv) * dlogp_dmean) - ddiff_grads_mean
            pi_grads_std = -((tf.expand_dims(BS_R, 1) - train_model.vf - train_model.cv) * dlogp_dstd) - ddiff_grads_std
            
            pg_grads = tf.gradients(train_model.mean, pi_params, grad_ys=pi_grads_mean) + tf.gradients(train_model.std_na, train_model.logstd_1a, grad_ys=pi_grads_std)
            
            cv_loss= tf.concat([tf.reshape(p, [-1]) for p in pg_grads], 0)
            cv_loss = tf.squeeze(tf.reduce_sum(tf.square(cv_loss)))       

            tf.summary.scalar("cv_loss", cv_loss)
            cv_params = [v for v in params if "cv" in v.name]
            cv_grads = tf.gradients(cv_loss , cv_params)
            cv_gradvars = list(zip(cv_grads, cv_params))
        else:
            pg_grads = tf.gradients(pg_loss, pi_params) +tf.gradients(pg_loss, train_model.logstd_1a)
      
        all_policy_grads = tf.concat([tf.reshape(pg, [-1]) for pg in pg_grads], 0)
        
        # policy gradients
        policy_gradvars = list(zip(pg_grads, pi_params + [train_model.logstd_1a]))
        vf_grads = tf.gradients(vf_loss, vf_params)
        vf_gradvars = list(zip(vf_grads, vf_params))
    
        grads_list = policy_gradvars + vf_gradvars
        if train_model.relaxed: grads_list += cv_gradvars
        for g, v in grads_list:
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name+"_grad", g)

        sum_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir)

        trainer = optim
        _train = trainer.apply_gradients(policy_gradvars)
  
        _vf_train = train_model.vf_optim.apply_gradients(vf_gradvars)
        
        self._step = 0
        
        def get_cv_grads(obs, old_actions, advs, rewards, bs_rewards, vf_in, values):
            advs = rewards - values
            td_map = {
                train_model.ob_no:obs, train_model.oldac_na:old_actions, train_model.X:vf_in,
                A: old_actions, ADV:advs, BS_R: bs_rewards
            }
            cv_gs = sess.run(cv_grads, td_map)
            return cv_gs
          
        def update_cv(mean_cv_gs):
            cv_gvs = list(zip(mean_cv_gs, cv_params))
            train_model.cv_optim.apply_gradients(cv_gvs)
            
        def update_policy_and_value(obs, old_actions, advs, rewards, bs_rewards, vf_in, values, summary=False):
            advs = rewards - values
            td_map = {
                train_model.ob_no:obs, train_model.oldac_na:old_actions, train_model.X:vf_in,
                A: old_actions, ADV:advs, R:rewards, BS_R: bs_rewards
            }
            if summary:
                sum_str, policy_loss, value_loss, _, = sess.run(
                    [sum_op, pg_loss, vf_loss, _train],
                    td_map
                )
                writer.add_summary(sum_str, self._step)
            else:
                policy_loss, value_loss, _ = sess.run(
                    [pg_loss, vf_loss, _train],
                    td_map
                )
            for _ in range(25): sess.run(_vf_train, 
                          {train_model.ob_no:obs, train_model.oldac_na:old_actions, train_model.X:vf_in, 
                           A: old_actions, ADV:advs, R:rewards, BS_R:bs_rewards})
            self._step += 1
            return policy_loss, value_loss
          
        def get_grads(obs, old_actions, advs, rewards, bs_rewards, vf_in, value):
            td_map = {
                train_model.ob_no:obs, train_model.oldac_na:old_actions, train_model.X:vf_in,
                A: old_actions, ADV:advs, R:rewards, BS_R:bs_rewards
            }
            _g = all_policy_grads / tf.to_float(tf.shape(rewards)[0])
            pg = sess.run(
                _g,
                td_map
            )
            return pg
          
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

        self.sess = sess
        self.get_cv_grads = get_cv_grads
        self.update_cv = update_cv
        self.update_policy_and_value = update_policy_and_value
        self.train_model = train_model
        self.step_model = step_model
        self.value = train_model.value
        self.get_grads = get_grads
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)
          
          
def pathlength(path):
    return path["reward"].shape[0]

class RolloutRunner(object):
    """
    Simulate the env and policy for max_pathlength steps
    """
    def __init__(self, env, policy, max_pathlength, gamma=0.99, lam=0.97, obfilter=None, animate=False, score=9100.0):
        self.env = env
        self.policy= policy
        self.max_pathlength = max_pathlength
        self.gamma = gamma
        self.lam = lam
        self.obfilter = obfilter
        self.animate = animate
        self._num_rollouts = 0
        self._num_steps = 0
        self.rewards = []
        self.episodes_till_done = 0
        self.frames_till_done = 0
        self.finished=False
        self.score=score

    def run(self, update_counters=True):
        ob = self.env.reset()
        prev_ob = np.float32(np.zeros(ob.shape))
        if self.obfilter: ob = self.obfilter(ob)
        terminated = False
    
        obs = []
        acs = []
        ac_dists = []
        logps = []
        rewards = []

        for _ in range(self.max_pathlength):
            if self.animate:
                self.env.render()
            state = np.concatenate([ob, prev_ob], -1)
            obs.append(state)
            ac, ac_dist, logp = self.policy.act(state)
            acs.append(ac)
            ac_dists.append(ac_dist)
            logps.append(logp)
            prev_ob = np.copy(ob)
            scaled_ac = self.env.action_space.low + (ac + 1.) * 0.5 * (self.env.action_space.high - self.env.action_space.low)
            scaled_ac = np.clip(scaled_ac, self.env.action_space.low, self.env.action_space.high)
            ob, rew, done, _ = self.env.step(scaled_ac)
            if self.obfilter: ob = self.obfilter(ob)
            rewards.append(rew)
            if done:
                terminated = True
                break
        self.rewards.append(sum(rewards))
        self.rewards = self.rewards[-100:]
        if update_counters:
            self._num_rollouts += 1
            self._num_steps += len(rewards)
              
        path = {"observation" : np.array(obs), "terminated" : terminated,
                "reward" : np.array(rewards), "action" : np.array(acs),
                "action_dist": np.array(ac_dists), "logp" : np.array(logps)}
        
        rew_t = path["reward"]
        vtarg = common.discount(rew_t, self.gamma)
        value = self.policy.predict(path["observation"], path)
        vpred_t = np.append(value, 0.0 if path["terminated"] else value[-1])
        delta_t = rew_t + self.gamma*vpred_t[1:] - vpred_t[:-1]
        bs_reward_t = common.discount(rew_t + self.gamma*vpred_t[1:], self.gamma) #bootstrapped reward
        adv_GAE = common.discount(delta_t, self.gamma * self.lam)
        
        if np.mean(self.rewards) >= self.score and not self.finished:
            self.episodes_till_done = self._num_rollouts
            self.frames_till_done = self._num_steps
            self.finished = True      
        
        return path, vtarg, value, adv_GAE, bs_reward_t


def learn(env, policy, seed, total_timesteps=int(10e6),
          max_grad_norm=0.5, p_lr=0.0001, vf_lr=0.001, cv_lr=0.001, cv_num=25, lrschedule='linear', epsilon=1e-5,
          alpha=0.99, gamma=0.99, lam=0.97, timesteps_per_batch=2500, num_timesteps=1e6,
          animate=False, callback=None, desired_kl=0.002, log_interval=100, logdir=None, endwhendone=False, score=9100.0, var_check=False):
  
    set_global_seeds(seed)
    num_procs = 1 

    obfilter = ZFilter(env.observation_space.shape)

    max_pathlength = env.spec.timestep_limit
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    stepsize = tf.Variable(np.float32(np.array(p_lr)), dtype=tf.float32)
    optim = tf.train.AdamOptimizer(stepsize)
    
    model = Model(optim=optim, policy=policy, ob_dim=ob_dim, ac_dim=ac_dim, num_procs=num_procs, 
                  max_grad_norm=max_grad_norm, vf_lr=vf_lr, cv_lr=cv_lr, cv_num=cv_num, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, logdir=logdir)
    
    runner = RolloutRunner(env, model.step_model, max_pathlength, gamma, lam, obfilter, animate, score)
    
    pi_var_list = []
    for var in tf.trainable_variables():
        if "pi" in var.name:
            pi_var_list.append(var)

    i = 0
    timesteps_so_far = 0
    while True:
        if timesteps_so_far > num_timesteps:
            break
        logger.log("********** Iteration %i ************"%i)
        
        if var_check:
            pgs = []
            if i %10 == 0:
                for _ in range(10):
                    path, vtarg, value, adv, bs_r = runner.run(update_counters=False)
                    
                    std_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    vf_in = model.step_model.preproc(path)
                    
                    pg = model.get_grads(path["observation"], path["action"], std_adv, vtarg, bs_r, vf_in, value)
                    pgs.append(pg)
                pgs = np.array(pg)
                pgv = np.var(pgs, axis=0)
                lpgv = np.log(pgv)
                lpgv = np.mean(lpgv)
                logger.record_tabular("log_variance", lpgv)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        vtargs = []
        advs = []
        std_advs = []
        vf_ins=[]
        bs_rewards = []
        values = []
        cv_grads = []
        while True:
            runner.animate = (len(paths)==0 and (i % 10 == 0) and animate)
            path, vtarg, value, adv, bs_r = runner.run()
            
            if model.train_model.relaxed:
                std_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                vf_in = model.step_model.preproc(path)
                cv_grad = model.get_cv_grads(path["observation"], path["action"], std_adv, vtarg, bs_r, vf_in, value)
                cv_grads.append(cv_grad)
                std_advs.append(std_adv)
                vf_ins.append(vf_in)
            vtargs.append(vtarg)
            values.append(value)
            advs.append(adv)            
            bs_rewards.append(bs_r)
            
            paths.append(path)
            n = pathlength(path)
            timesteps_this_batch += n
            timesteps_so_far += n
            if timesteps_this_batch > timesteps_per_batch:
                break
            
        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        action_na = np.concatenate([path["action"] for path in paths])
        oldac_dist = np.concatenate([path["action_dist"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        rewards_n = np.concatenate(vtargs)
        values_n = np.concatenate(values)
        bs_rewards_n = np.concatenate(bs_rewards) 
        
        # for value function 
        x = np.concatenate([model.step_model.preproc(p) for p in paths])
        logger.record_tabular("EVBefore", common.explained_variance(model.step_model.value(ob_no, x), rewards_n))
            
        # update policy and value network
        policy_loss, value_loss = model.update_policy_and_value(ob_no, action_na, standardized_adv_n, rewards_n, bs_rewards_n, x, values_n)
        
        if model.train_model.relaxed:
            # update control variate cv_num times
            for r in range(cv_num):
                cv_gs = []            
                for k in range(len(cv_grads[0])):
                    cvg = 0
                    for l in range(len(cv_grads)):
                        cvg += cv_grads[l][k]
                    cvg /= len(cv_grads)
                    cv_gs.append(cvg.astype(np.float32))
                
                model.update_cv(cv_gs)
                
                # get updated cv_grads
                cv_grads=[]
                for p in range(len(paths)):
                    cv_grads.append(model.get_cv_grads(paths[p]["observation"], paths[p]["action"], std_advs[p], vtargs[p], bs_rewards[p], vf_ins[p], values[p]))            
        
        logger.record_tabular("EVAfter", common.explained_variance(model.step_model.value(ob_no, x), rewards_n))

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)
        # Adjust stepsize
        kl = model.step_model.compute_kl(ob_no, oldac_dist)
        if kl > desired_kl * 2:
            logger.log("kl too high")
            model.sess.run(tf.assign(stepsize, tf.maximum(min_stepsize, stepsize / 1.5)))
        elif kl < desired_kl / 2:
            logger.log("kl too low")
            model.sess.run(tf.assign(stepsize, tf.minimum(max_stepsize, stepsize * 1.5)))
        else:
            logger.log("kl just right!")
        
        logger.record_tabular("last_r", runner.rewards[-1])
        logger.record_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("EpRewSEM", np.std([path["reward"].sum()/np.sqrt(len(paths)) for path in paths]))
        logger.record_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logger.record_tabular("KL", kl)
        if callback:
            callback()
        logger.dump_tabular()
        i += 1
    with open(logdir+"/results.txt", "w") as f: 
        f.write("-----------------------------\n")
        f.write("Done!\n")
        f.write("episodes till done: %s\n" %runner.episodes_till_done)
        f.write("frames till done: %s\n" %runner.frames_till_done)
        f.write("-----------------------------") 
