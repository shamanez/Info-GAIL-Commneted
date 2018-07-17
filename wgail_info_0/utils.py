import numpy as np
import cv2
import random
import time
import tensorflow as tf
import scipy.signal
from keras.applications.resnet50 import preprocess_input

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import keras.backend as K

from collections import deque
import pdb


seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32


def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def gauss_prob_val(mu, logstd, x):
    std = np.exp(logstd)
    var = np.square(std)
    gp = np.exp(-np.square(x - mu)/(2*var)) / ((2*np.pi)**.5 * std)
    return np.prod(gp, axis=1)

def gauss_prob(mu, logstd, x):
    std = tf.exp(logstd)
    var = tf.square(std)
    gp = tf.exp(-tf.square(x - mu)/(2*var)) / ((2*np.pi)**.5 * std)
    return tf.reduce_prod(gp, [1])

def gauss_log_prob(mu, logstd, x):

    var = tf.exp(2*logstd)
    gp = -tf.square(x - mu)/(2 * var) - .5*tf.log(tf.constant(2*np.pi)) - logstd
    return tf.reduce_sum(gp, [1])  #get he gassian log probability

def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd
    return gauss_KL(mu1, logstd1, mu2, logstd2)

def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2*logstd1)
    var2 = tf.exp(2*logstd2)
    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2))/(2*var2) - 0.5)
    return kl

def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5*np.log(2*np.pi*np.e), tf.float32))
    return h

def gauss_sample(mu, logstd):
    return mu + tf.exp(logstd)*tf.random_normal(tf.shape(logstd))

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
            "shape function assumes that shape is fully known"
    return out

def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):

    
    grads = tf.gradients(loss, var_list)  #get the gradients acordin to the losses 
    K=[tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)]
    return tf.concat(axis=0, values=[tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)])
    return K

def get_feat(imgs, feat_extractor):
    x = preprocess_input(imgs.astype(np.float32))
    x = feat_extractor.predict(x)
    return x

def get_state(ob, feat_dim, aux_dim, feat_extractor):
  
    img = ob.img
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
    img = cv2.resize(img, (200, 150))
    img = img[40:, :, :]
    x = np.expand_dims(img, axis=0)
    feat = get_feat(x, feat_extractor) #get the resnet featires

    aux = np.zeros(aux_dim, dtype=np.float32)
    aux[0] = ob.damage
    aux[1] = ob.speedX
    aux[2] = ob.speedY
    aux[3] = ob.speedZ
    aux[4:7] = ob.pre_action_0
    aux[7:10] = ob.pre_action_1

    img = (np.array(cv2.resize(img, (50, 50)), dtype=np.float32) - 128.) / 128.
    return feat, np.expand_dims(aux, axis=0), np.expand_dims(img, axis=0),

def rollout_contin(env, agent, feat_extractor, feat_dim, aux_dim, encode_dim,
                   max_step_limit, min_step_limit, pre_step, paths_per_collect,
                   pre_actions, discriminate, posterior):
    paths = []
    timesteps_sofar = 0
    encode_axis = 0
    
    for p in xrange(2):#paths_per_collect): #This will run to collect paths #first iteration 30 rollouts after that 19
        print "Rollout index:", p
        feats, auxs, encodes, imgs, raws, actions, logstds = \
                [], [], [], [], [], [], []
        feat, aux, img = get_state(env.reset(relaunch=True), feat_dim, aux_dim,   #getting the state from the  simulator
                                   feat_extractor)


        encode = np.zeros((1, encode_dim), dtype=np.float32)  #[0. 0.] 
        encode[0, encode_axis] = 1
        encode_axis = (encode_axis + 1) % encode_dim
        print "Encode:", encode   #[1. 0.] latent vector   #econdong is change is each roll out either [1,0 ] or [0,1] (This is the latenet stucture)
        reward_d = 0
        reward_p = 0
        
        for i in xrange(max_step_limit):
            if i < min_step_limit:
                print("My iteation is :",i)
                print("My Actions are always [000].......")
                action = np.zeros(3, dtype=np.float32)
             
                res = env.step(action)  #go ttake a random acton
            
          
            elif i < min_step_limit + pre_step:
                print("I am executing the given actions from thepre actions.npz")
                print("My iteration is :",i)
                action = pre_actions[i - min_step_limit]
                res = env.step(action)
                if i == min_step_limit + pre_step - 1:
                    feat, aux, img = get_state(res[0], feat_dim, aux_dim,   #getting the state representaion of the curent featrues state and auxilary variables
                                               feat_extractor)
                    
                   
            else:
                print("now after the pre actions I my agent will start executing actions")
                feats.append(feat) #initail featrures 
                auxs.append(aux)
                encodes.append(encode)   #this one doe not change it is about the latent variables

            
                
                if i < 230:
                    logstd = np.array([[-2.8, -3.5, -4]], dtype=np.float32)

                else:
                    logstd = np.array([[-3.5, -4, -5]], dtype=np.float32)
                logstds.append(logstd)
        
                action = agent.act(feat, aux, encode, logstd) #get the action from the generator
                actions.append(action)
                imgs.append(img)
                raw = res[1] #this is a reward funtion by the agent.act function. this is -200 if collided, go backwords or out_of track (These stages are terminas)
                raws.append(raw)



                reward_d += discriminate.predict([img, aux, action])[0, 0] * 0.1       #this is the discriminator reward This actually continumasy sum up the reward signal for each state and aciond
                #print(posterior.predict([img, aux, action]))
                #print(encode)
                #print(np.log(posterior.predict([img, aux, action])))
                reward_p += np.sum(np.log(posterior.predict([img, aux, action]))
                                   * encode)

                #print(reward_p)
                #print(reward_d)
              

                if res[2] or i + 1 == max_step_limit:
                    path = dict2(feats = np.concatenate(feats),             #single path contains of these kind of dictionary and data here my testing code has 30
                                 auxs = np.concatenate(auxs),
                                 encodes = np.concatenate(encodes),
                                 imgs = np.concatenate(imgs),
                                 actions = np.concatenate(actions),
                                 logstds = np.concatenate(logstds),
                                 raws = np.array(raws))
                    paths.append(path)
                    step = i + 1 - min_step_limit - pre_step
                    print "Step:", step, "Reward_d:", reward_d, \
                        "Reward_p:", reward_p, "Total:", reward_d + reward_p + 2 * step
                    break

                res = env.step(action[0])
                feat, aux, img = get_state(res[0], feat_dim, aux_dim, feat_extractor)


    env.end()
    return paths


'''


class LinearBaseline(object):
    coeffs = None

    def _features(self, path):
        o = path["states"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + lamb * np.identity(n_col),
            featmat.T.dot(returns))[0]

    def predict(self, path):
        return np.zeros(len(path["rewards"])) if self.coeffs is None else \
                self._features(path).dot(self.coeffs)
'''


def pathlength(path):
    return len(path["actions"])

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

'''
class TimeDependentBaseline(object):
    def __init__(self):
        self.baseline = None

    def fit(self, paths):
        rets = [path["returns"] for path in paths]
        maxlen = max(len(ret) for ret in rets)
        retsum = np.zeros(maxlen)
        retcount = np.zeros(maxlen)
        for ret in rets:
            retsum[:len(ret)] += ret
            retcount[:len(ret)] += 1
        retmean = retsum / retcount
        self.baseline = retmean
        pred = np.concatenate([self.predict(path) for path in paths])
        return {"EV" : explained_variance(pred, np.concatenate(rets))}

    def predict(self, path):
        if self.baseline is None:
            return np.zeros(pathlength(path))
        else:
            lenpath = pathlength(path)
            lenbase = len(self.baseline)
            if lenpath > lenbase:
                return np.concatenate([self.baseline, self.baseline[-1] +
                                       np.zeros(lenpath-lenbase)])
            else:
                return self.baseline[:lenpath]
'''


class NNBaseline(object):
    def __init__(self, sess, feat_dim, aux_dim, encode_dim, lr_baseline,
                 b_iter, batch_size):
        print "Now we build baseline"
        self.model = self.create_net(feat_dim, aux_dim, encode_dim, lr_baseline)
        self.sess = sess
        self.b_iter = b_iter
        self.batch_size = batch_size
        self.first_time = True
        self.mixfrac = 0.1

    def create_net(self, feat_dim, aux_dim, encode_dim, lr_baseline):
        feats = Input(shape=[feat_dim[0], feat_dim[1], feat_dim[2]])
        x = Convolution2D(256, 3, 3)(feats)
        x = LeakyReLU()(x)
        x = Convolution2D(256, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        auxs = Input(shape=[aux_dim + 1])
        encodes = Input(shape=[encode_dim])
        h = merge([x, auxs, encodes], mode='concat')
        h = Dense(256)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        p = Dense(1)(h)

        model = Model(input=[feats, auxs, encodes], output=p)
        adam = Adam(lr=lr_baseline)
        model.compile(loss='mse', optimizer=adam)
        return model

    def _get_aux(self, path):
        # Add time stamp
        aux = path["auxs"].astype('float32')
        aux = aux.reshape(aux.shape[0], -1)
        l = len(path["auxs"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([aux, al], axis=1)

    def fit(self, paths):
        feats = np.concatenate([path["feats"] for path in paths])
        auxs = np.concatenate([self._get_aux(path) for path in paths])
        encodes = np.concatenate([path["encodes"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])



        if self.first_time:
            print("I AM INSIDE THIS")
            self.first_time = False
            b_iter = 100
        else:
            print("2@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            returns_old = np.concatenate([self.predict(path) for path in paths])
            returns = returns * self.mixfrac + returns_old * (1 - self.mixfrac)
            b_iter = self.b_iter

        num_data = feats.shape[0]
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        train_val_ratio = 0.7
        num_train = int(num_data * train_val_ratio)
        feats_train = feats[idx][:num_train]
        auxs_train = auxs[idx][:num_train]
        encodes_train = encodes[idx][:num_train]
        returns_train = returns[idx][:num_train]

        feats_val = feats[idx][num_train:]
        auxs_val = auxs[idx][num_train:]
        encodes_val = encodes[idx][num_train:]
        returns_val = returns[idx][num_train:]

        print(returns_train)

        start = 0
        batch_size = self.batch_size
        for i in xrange(b_iter):
            loss = self.model.train_on_batch(
                [feats_train[start:start + batch_size],
                 auxs_train[start:start + batch_size],
                 encodes_train[start:start + batch_size]],
                returns_train[start:start + batch_size]

            )

          
            start += batch_size
            if start >= num_train:
                start = (start + batch_size) % num_train
            val_loss = np.average(np.square(self.model.predict(
                [feats_val, auxs_val, encodes_val]).flatten() - returns_val))
            print "Baseline step:", i, "loss:", loss, "val:", val_loss

    def predict(self, path):

        if self.first_time:
            print("Miyaaaaaaaaaaaaaaaaaaaaw")
            return np.zeros(pathlength(path))  #zeros equal to lengthe of actions 
        else:
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            ret = self.model.predict(
                [path["feats"], self._get_aux(path), path["encodes"]])
        return np.reshape(ret, (ret.shape[0], ))



class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)


class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(v, tf.reshape(theta[start:start + size], shape))
            )
            start += size
        self.op = tf.group(*assigns)  #assigning varaibels vack to their orgianal shape ?? 

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in xrange(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size #this is the bffe sze
        self.num_paths = 0
        self.buffer = deque()

    def get_sample(self, sample_size):
        if self.num_paths < sample_size:
            return random.sample(self.buffer, self.num_paths)
        else:
            return random.sample(self.buffer, sample_size)

    def size(self):
        return self.buffer_size

    def add(self, path):
        if self.num_paths < self.buffer_size:
            self.buffer.append(path)  #add to the buffer
            self.num_paths += 1
        else:
            self.buffer.popleft() #left buffering
            self.buffer.append(path)

    def count(self):
        return self.num_paths

    def erase(self):
        self.buffer = deque()
        self.num_paths = 0
