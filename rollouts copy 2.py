from collections import deque, defaultdict

import numpy as np
from mpi4py import MPI
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from recorder import Recorder
import math
from patience import Patience
from utils import small_convnet, flatten_two_dims, unflatten_first_dim, getsess, unet
@tf.function
def train_step(model, s,ac, labels):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    
    loss_object = tf.keras.losses.MSE()
    optimizer = tf.keras.optimizers.Adam()
    
    with tf.GradientTape() as tape:
        predictions = model(s,ac)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, s,ac, labels):
    loss_object = tf.keras.losses.MSE()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    predictions = model(s,ac)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

class Rollout(object):
    def __init__(self, ob_space, ac_space, nenvs, nsteps_per_seg, nsegs_per_env, nlumps, envs, policy,
                 int_rew_coeff, ext_rew_coeff, record_rollouts, dynamics,patience):
        self.nenvs = nenvs
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.nlumps = nlumps
        self.lump_stride = nenvs // self.nlumps
        self.envs = envs
        self.policy = policy
        self.dynamics = dynamics
        self.patience = patience
        self.reward_fun = lambda ext_rew, int_rew: ext_rew_coeff * np.clip(ext_rew, -1., 1.) + int_rew_coeff * int_rew

        self.buf_vpreds = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_nlps = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_ext_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_acs = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        self.buf_obs = np.empty((nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype)
        self.buf_obs_last = np.empty((nenvs, self.nsegs_per_env, *self.ob_space.shape), np.float32)

        self.buf_news = np.zeros((nenvs, self.nsteps), np.float32)
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_last = self.buf_vpreds[:, 0, ...].copy()

        self.env_results = [None] * self.nlumps
        # self.prev_feat = [None for _ in range(self.nlumps)]
        # self.prev_acs = [None for _ in range(self.nlumps)]
        self.int_rew = np.zeros((nenvs,), np.float32)

        self.recorder = Recorder(nenvs=self.nenvs, nlumps=self.nlumps) if record_rollouts else None
        self.statlists = defaultdict(lambda: deque([], maxlen=100))
        self.stats = defaultdict(float)
        self.best_ext_ret = None
        self.all_visited_rooms = []
        self.all_scores = []

        self.step_count = 0

        #########################################################
        self.ps_buf = self.int_rew = np.zeros((nenvs,), np.float32)
        self.ps_buf_nold = self.int_rew = np.zeros((nenvs,), np.float32)
        self.ac_buf = self.int_rew = np.zeros((nenvs,), np.float32)
        self.buf_acs_nold = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        self.ac_buf_nold = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        self.nthe = 2
        self.ac_list = []
        self.ps_list = []
        self.rew_list = []
        self.obs_list = []
        self.feat_list = []
        self.patience_pred = 0
        self.pat = 0
        self.mean = np.zeros((nenvs,), np.float32)
        self.var = np.ones_like((nenvs,), np.float32)
        #########################################################
    def collect_rollout(self):
        self.ep_infos_new = []
        for t in range(self.nsteps):
            self.rollout_step()
        self.calculate_reward()
        self.update_info()

    def calculate_reward(self):
        #????????? ??????
        ##############################################################################
        int_rew, ps, ac, feat = self.dynamics.calculate_loss(ob=self.buf_obs,
                                               last_ob=self.buf_obs_last,
                                               acs=self.buf_acs, feat_input = [],pat = 0)
        print(int_rew.shape)
        self.ac_list.append(ac)
        self.ps_list.append(ps)
        self.rew_list.append(int_rew)
        self.obs_list.append(self.buf_obs)
        self.feat_list.append(feat)
        #self.feat_list = tf.stack([self.feat_list, feat])
        #print(self.feat_list[0])
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        """
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        """
        
        if len(self.ac_list) > (self.nthe):
            self.ac_list = self.ac_list[1:]
            self.ps_list = self.ps_list[1:]
            self.rew_list = self.rew_list[1:]
            self.obs_list = self.obs_list[1:]
        
        if self.step_count/self.nsteps > self.nthe:
            #ac_spec = tf.one_hot(self.ac_list[0], 4, axis=2)
            #ac_spec = flatten_two_dims(ac_spec)
            #print("ac_speccccccccccccccccccc:",ac_spec.shape)
            int_rew_now, ps_now, ac_now, feat_now = self.dynamics.calculate_loss(ob=self.obs_list[0],
                                                last_ob=self.buf_obs_last,
                                                acs=self.ac_list[0],feat_input = self.feat_list[0], pat = 1)              #ps_buf_nold S_1**
            
            print("Sucess///////////////////////////////////////////////////////////////")
            sess = tf.Session()
            
            coc = self.rew_list[0] - int_rew_now
            print("coc",coc)
            norm_coc = np.multiply((coc - self.mean),np.reciprocal(np.sqrt(self.var)))
            
            self.mean = 0.9 * self.mean + 0.1 * coc
            self.var = 0.9 * self.var + 0.1 * coc**2
            
            node = tf.math.sigmoid(norm_coc)
            
            self.pat = sess.run(node)
            print("int_rew_now",int_rew_now.shape)
            print("patshape",self.pat.shape)
            #???????????? sigmoid function???????????? 0?????? 1????????? ??? ????????????.
            #diff = sess.run(node)
            #MontezumaRevengeNoFrameskip-v4
            #self.patience = 0.9 * self.patience + 0.1 * diff
            #########################
            #self.patience.train_step()
            # patience_pred 
            #model = Patience(self.obs_list[0],self.buf_obs_last,self.ac_list[0],self.feat_list[0])
            #self.patience_pred = Patience(state = self.obs_list[0],action = self.ac_list[0], labels = self.pat, ac_space = self.ac_space, auxiliary_task = self.dynamics.auxiliary_task).get_loss()
            #print("hi",self.feat_list[0].shape,self.ac_list[0].shape)
            #self.patience_pred = model(self.feat_list[0],self.ac_list[0])
            pat_pred = self.patience.calculate_loss(ob=self.obs_list[0],
                                                last_ob=self.buf_obs_last,
                                                acs=self.ac_list[0],feat_input = self.feat_list[0], pat = 1)
            
            # model.calculate_loss(ob=self.obs_list[0],
            #                                    last_ob=self.buf_obs_last,
            #                                    acs=self.ac_list[0], feat_input = self.feat_list[0])
            #train_step(model, self.feat_list[0],self.ac_list[0],self.pat)
            #########################
            #int_rew = int_rew * self.patience_pred
            # elementwise multiply
            #int_rew = np.multiply(int_rew, self.patience_pred)
            
        ################################################################################
        self.buf_rews[:] = self.reward_fun(int_rew=int_rew, ext_rew=self.buf_ext_rews)
        

    def rollout_step(self):
        t = self.step_count % self.nsteps
        s = t % self.nsteps_per_seg
        for l in range(self.nlumps):
            obs, prevrews, news, infos = self.env_get(l)
            # if t > 0:
            #     prev_feat = self.prev_feat[l]
            #     prev_acs = self.prev_acs[l]
            for info in infos:
                epinfo = info.get('episode', {})
                mzepinfo = info.get('mz_episode', {})
                retroepinfo = info.get('retro_episode', {})
                epinfo.update(mzepinfo)
                epinfo.update(retroepinfo)
                if epinfo:
                    if "n_states_visited" in info:
                        epinfo["n_states_visited"] = info["n_states_visited"]
                        epinfo["states_visited"] = info["states_visited"]
                    self.ep_infos_new.append((self.step_count, epinfo))

            sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)

            acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)
            self.env_step(l, acs)

            # self.prev_feat[l] = dyn_feat
            # self.prev_acs[l] = acs
            self.buf_obs[sli, t] = obs
            self.buf_news[sli, t] = news
            self.buf_vpreds[sli, t] = vpreds
            self.buf_nlps[sli, t] = nlps
            self.buf_acs[sli, t] = acs
            if t > 0:
                self.buf_ext_rews[sli, t - 1] = prevrews
            # if t > 0:
            #     dyn_logp = self.policy.call_reward(prev_feat, pol_feat, prev_acs)
            #
            #     int_rew = dyn_logp.reshape(-1, )
            #
            #     self.int_rew[sli] = int_rew
            #     self.buf_rews[sli, t - 1] = self.reward_fun(ext_rew=prevrews, int_rew=int_rew)
            if self.recorder is not None:
                self.recorder.record(timestep=self.step_count, lump=l, acs=acs, infos=infos, int_rew=self.int_rew[sli],
                                     ext_rew=prevrews, news=news)
        self.step_count += 1
        if s == self.nsteps_per_seg - 1:
            for l in range(self.nlumps):
                sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)
                nextobs, ext_rews, nextnews, _ = self.env_get(l)
                self.buf_obs_last[sli, t // self.nsteps_per_seg] = nextobs
                if t == self.nsteps - 1:
                    self.buf_new_last[sli] = nextnews
                    self.buf_ext_rews[sli, t] = ext_rews
                    _, self.buf_vpred_last[sli], _ = self.policy.get_ac_value_nlp(nextobs)
                    # dyn_logp = self.policy.call_reward(self.prev_feat[l], last_pol_feat, prev_acs)
                    # dyn_logp = dyn_logp.reshape(-1, )
                    # int_rew = dyn_logp
                    #
                    # self.int_rew[sli] = int_rew
                    # self.buf_rews[sli, t] = self.reward_fun(ext_rew=ext_rews, int_rew=int_rew)

    def update_info(self):
        all_ep_infos = MPI.COMM_WORLD.allgather(self.ep_infos_new)
        all_ep_infos = sorted(sum(all_ep_infos, []), key=lambda x: x[0])
        if all_ep_infos:
            all_ep_infos = [i_[1] for i_ in all_ep_infos]  # remove the step_count
            keys_ = all_ep_infos[0].keys()
            all_ep_infos = {k: [i[k] for i in all_ep_infos] for k in keys_}

            self.statlists['eprew'].extend(all_ep_infos['r'])
            self.stats['eprew_recent'] = np.mean(all_ep_infos['r'])
            self.statlists['eplen'].extend(all_ep_infos['l'])
            self.stats['epcount'] += len(all_ep_infos['l'])
            self.stats['tcount'] += sum(all_ep_infos['l'])
            if 'visited_rooms' in keys_:
                # Montezuma specific logging.
                self.stats['visited_rooms'] = sorted(list(set.union(*all_ep_infos['visited_rooms'])))
                self.stats['pos_count'] = np.mean(all_ep_infos['pos_count'])
                self.all_visited_rooms.extend(self.stats['visited_rooms'])
                self.all_scores.extend(all_ep_infos["r"])
                self.all_scores = sorted(list(set(self.all_scores)))
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited rooms")
                    print(self.all_visited_rooms)
                    print("All scores")
                    print(self.all_scores)
            if 'levels' in keys_:
                # Retro logging
                temp = sorted(list(set.union(*all_ep_infos['levels'])))
                self.all_visited_rooms.extend(temp)
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited levels")
                    print(self.all_visited_rooms)

            current_max = np.max(all_ep_infos['r'])
        else:
            current_max = None
        self.ep_infos_new = []

        if current_max is not None:
            if (self.best_ext_ret is None) or (current_max > self.best_ext_ret):
                self.best_ext_ret = current_max
        self.current_max = current_max

    def env_step(self, l, acs):
        self.envs[l].step_async(acs)
        self.env_results[l] = None

    def env_get(self, l):
        if self.step_count == 0:
            ob = self.envs[l].reset()
            out = self.env_results[l] = (ob, None, np.ones(self.lump_stride, bool), {})
        else:
            if self.env_results[l] is None:
                out = self.env_results[l] = self.envs[l].step_wait()
            else:
                out = self.env_results[l]
        return out
