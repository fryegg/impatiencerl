import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from auxiliary_tasks import JustPixels
from utils import small_convnet, flatten_two_dims, unflatten_first_dim, getsess, unet


class Dynamics(object):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='dynamics'):
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.obs = self.auxiliary_task.obs
        self.last_ob = self.auxiliary_task.last_ob
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        #############################################
        # 여기는 수정부분

        #############################################
        if predict_from_pixels:
            self.features = self.get_features(self.obs, reuse=False) #s_t
        else:
            self.features = tf.stop_gradient(self.auxiliary_task.features) #s_t

        self.out_features = self.auxiliary_task.next_features # find s_t+1

        with tf.variable_scope(self.scope + "_loss"):
            self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        
        def add_ac(x):
            return tf.concat([x, ac], axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x = tf.reshape(x,(-1,512))
            x = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)

            def residual(x):
                res = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
                res = tf.layers.dense(add_ac(res), self.hidsize, activation=None)
                return x + res

            for _ in range(4):
                x = residual(x)
            n_out_features = self.out_features.get_shape()[-1].value
            x = tf.layers.dense(add_ac(x), n_out_features, activation=None)
            x = unflatten_first_dim(x, sh)
            #####################################################
            #ps = (tf.reduce_mean(tf.stop_gradient(self.out_features), -1))
            ps = tf.reduce_mean(x,-1)
            #print("reward: ", tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1).shape)
            #####################################################
            # tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1), buf_ac
        return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1), ps, self.features # 84 x 84 x 128 x128 int x: state prediction - non update next obs RMS -> reward : 128(pararellel thread) x 128(rollouts length)
        # return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1), ps, self.features

    def calculate_loss(self, ob, last_ob, acs, feat_input, pat):
        n_chunks = 8
        ans_buf = []
        ans = []
        ac_buf = []
        ps_buf = []
        feat_buf = []
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        # important last_ob: pi(s_t+1) obs:pi(s_t) ac: (s0,a0)
        for i in range(n_chunks):
            if pat:
                (ans, ps, feat) = getsess().run(self.loss,{self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                              self.ac: acs[sli(i)], self.features:feat_input[sli(i)]})
                ac = tf.one_hot(acs[sli(i)], 4, axis=2)
                sh = tf.shape(ac)
                ac = flatten_two_dims(ac)
                print("ac.shape:",ac.shape)
                print("feat.shape:",feat.shape)
                #print("result: ", tf.concat([feat, ac], axis=-1).shape)
            else:
                (ans, ps, feat) = getsess().run(self.loss,{self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)], self.ac: acs[sli(i)]})
                ac = tf.one_hot(acs[sli(i)], 4, axis=2)
                sh = tf.shape(ac)
                ac = flatten_two_dims(ac)
                print("ac.shape:",ac.shape)
                print("feat.shape:",feat.shape)
                #print("result: ", tf.concat([feat, ac], axis=-1).shape)
            ans_buf.append(ans)
            ps_buf.append(ps)
            ac_buf.append(acs[sli(i)])
            feat_buf.append(feat)
        return np.concatenate(ans_buf,0), np.concatenate(ps_buf,0), np.concatenate(ac_buf,0), np.concatenate(feat_buf,0)
        
class UNet(Dynamics):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='pixel_dynamics'):
        assert isinstance(auxiliary_task, JustPixels)
        assert not predict_from_pixels, "predict from pixels must be False, it's set up to predict from features that are normalized pixels."
        super(UNet, self).__init__(auxiliary_task=auxiliary_task,
                                   predict_from_pixels=predict_from_pixels,
                                   feat_dim=feat_dim,
                                   scope=scope)

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        nl = tf.nn.leaky_relu
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        ac_four_dim = tf.expand_dims(tf.expand_dims(ac, 1), 1)
        def add_ac(x):
            if x.get_shape().ndims == 2:
                return tf.concat([x, ac], axis=-1)
            elif x.get_shape().ndims == 4:
                sh = tf.shape(x)
                return tf.concat(
                    [x, ac_four_dim + tf.zeros([sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value], tf.float32)],
                    axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
            x = unflatten_first_dim(x, sh)
        self.prediction_pixels = x * self.ob_std + self.ob_mean
        return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, [2, 3, 4])
