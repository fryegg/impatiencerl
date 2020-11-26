import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from auxiliary_tasks import JustPixels
from utils import small_convnet, flatten_two_dims, unflatten_first_dim, getsess, unet
from tensorflow.keras import Model

class Patience(Model):
    def __init__(self):
        super(Patience, self).__init__()
    def call(self, s, ac):
        def residual(x,ac):
            res = tf.layers.dense(tf.concat([x, ac], axis=-1), 4, activation=tf.nn.leaky_relu)
            res = tf.layers.dense(tf.concat([res, ac], axis=-1), 4, activation=None)
            return x + res
        ac = tf.one_hot(ac, 4, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        for _ in range(4):
            s = residual(s,ac)
        s = tf.layers.dense(tf.concat([s, ac], axis=-1), 1, activation=None)
        return s
        
class Patience2(object):
    def __init__(self, auxiliary_task, state, action, ac_space, labels, scope='patience'):
        self.ac_space = ac_space
        self.state = state
        self.action = action
        self.labels = labels
        self.optimizer = tf.keras.optimizers.Adam()
        self.scope = scope
        self.hidsize = auxiliary_task.hidsize
    def get_loss(self):
        ac = tf.one_hot(self.action, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)

        def add_ac(x):
            return tf.concat([x, ac], axis=-1)

        with tf.variable_scope(self.scope):
            #x = flatten_two_dims(self.state)
            x = self.state
            x = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)

            def residual(x):
                res = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
                res = tf.layers.dense(add_ac(res), self.hidsize, activation=None)
                return x + res

            for _ in range(4):
                x = residual(x)
            x = tf.layers.dense(add_ac(x), 1, activation=None)
        return x