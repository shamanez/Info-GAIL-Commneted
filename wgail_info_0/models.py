from utils import *
import numpy as np
import time
import math
import argparse
from keras.initializers import normal, identity, uniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import keras.backend as K
import json

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.utils.np_utils import to_categorical
import pdb


parser = argparse.ArgumentParser(description="TRPO")
parser.add_argument("--paths_per_collect", type=int, default=10)
parser.add_argument("--max_step_limit", type=int, default=300)
parser.add_argument("--min_step_limit", type=int, default=100)
parser.add_argument("--pre_step", type=int, default=100)
parser.add_argument("--n_iter", type=int, default=1000)
parser.add_argument("--gamma", type=float, default=.95)
parser.add_argument("--lam", type=float, default=.97)
parser.add_argument("--max_kl", type=float, default=0.01)
parser.add_argument("--cg_damping", type=float, default=0.1)
parser.add_argument("--lr_discriminator", type=float, default=5e-5)
parser.add_argument("--d_iter", type=int, default=100)
parser.add_argument("--clamp_lower", type=float, default=-0.01)
parser.add_argument("--clamp_upper", type=float, default=0.01)
parser.add_argument("--lr_baseline", type=float, default=1e-4)
parser.add_argument("--b_iter", type=int, default=25)
parser.add_argument("--lr_posterior", type=float, default=1e-4)
parser.add_argument("--p_iter", type=int, default=50)
parser.add_argument("--buffer_size", type=int, default=75)
parser.add_argument("--sample_size", type=int, default=2) #previously this was 50 now I am changing this to 3
parser.add_argument("--batch_size", type=int, default=10)

args = parser.parse_args()


class TRPOAgent(object):
    print("Creating the TRPOAgent")
    
    config = dict2(paths_per_collect = args.paths_per_collect,
                   max_step_limit = args.max_step_limit,
                   min_step_limit = args.min_step_limit,
                   pre_step = args.pre_step,
                   n_iter = args.n_iter,
                   gamma = args.gamma,
                   lam = args.lam,
                   max_kl = args.max_kl,
                   cg_damping = args.cg_damping,
                   lr_discriminator = args.lr_discriminator,
                   d_iter = args.d_iter,
                   clamp_lower = args.clamp_lower,
                   clamp_upper = args.clamp_upper,
                   lr_baseline = args.lr_baseline,
                   b_iter = args.b_iter,
                   lr_posterior = args.lr_posterior,
                   p_iter = args.p_iter,
                   buffer_size = args.buffer_size,
                   sample_size = args.sample_size,
                   batch_size = args.batch_size)

    def __init__(self, env, sess, feat_dim, aux_dim, encode_dim, action_dim,
                 img_dim, pre_actions):
        self.env = env
        self.sess = sess
        self.buffer = ReplayBuffer(self.config.buffer_size)  #buffer size is 75
        self.feat_dim = feat_dim #[7, 13, 1024] resnet layer etraction
        self.aux_dim = aux_dim #10 given in the paper making it easy to 
        self.encode_dim = encode_dim # 1 or zero
        self.action_dim = action_dim #three dims aceleration ,stering and breaks()
        self.img_dim = img_dim #[50, 50, 3] input to the discriminator
        self.pre_actions = pre_actions #200 actions pre-taken

        #Placeholders

        self.feats = feats = tf.placeholder(                   
            dtype, shape=[None, feat_dim[0], feat_dim[1], feat_dim[2]]  #[7, 13, 1024] like tensor  here the output is 7*3 times 1024(bumber of channels)
        )
        self.auxs = auxs = tf.placeholder(dtype, shape=[None, aux_dim]) #place holdets for auxilary
        self.encodes = encodes = tf.placeholder(dtype, shape=[None, encode_dim])
        self.actions = actions = tf.placeholder(dtype, shape=[None, action_dim]) #this is for the pre-action list

       

        self.advants = advants = tf.placeholder(dtype, shape=[None]) #these are advantage units which usese to optimize the genertor using TRPO
        self.oldaction_dist_mu = oldaction_dist_mu = \
                tf.placeholder(dtype, shape=[None, action_dim])  #mean of the old actoon ditribution
        self.oldaction_dist_logstd = oldaction_dist_logstd = \
                tf.placeholder(dtype, shape=[None, action_dim])   #This is to calculate the old action standard deviation

        # Create neural network.
        print "Now we build trpo generator"
        self.generator = self.create_generator(feats, auxs, encodes)

        
        print "Now we build discriminator"
        self.discriminator, self.discriminate = \
            self.create_discriminator(img_dim, aux_dim, action_dim)

        
        print "Now we build posterior"
        self.posterior = \
            self.create_posterior(img_dim, aux_dim, action_dim, encode_dim)
        self.posterior_target = \
            self.create_posterior(img_dim, aux_dim, action_dim, encode_dim)

        self.demo_idx = 0

        action_dist_mu = self.generator.outputs[0]  #batch_size * number of actions
     

        
        # self.action_dist_logstd_param = action_dist_logstd_param = \
        #         tf.placeholder(dtype, shape=[1, action_dim])
        # action_dist_logstd = tf.tile(action_dist_logstd_param,
        #                              tf.pack((tf.shape(action_dist_mu)[0], 1)))
        action_dist_logstd = tf.placeholder(dtype, shape=[None, action_dim])  #this is to get the standard diciatopn of each action

        eps = 1e-8

        self.action_dist_mu = action_dist_mu
        self.action_dist_logstd = action_dist_logstd

        N = tf.shape(feats)[0] #[7, 13, 1024] so this is 7
       
        # compute probabilities of current actions and old actions
        log_p_n = gauss_log_prob(action_dist_mu, action_dist_logstd, actions) #this takes action discribution from the model 
        log_oldp_n = gauss_log_prob(oldaction_dist_mu, oldaction_dist_logstd, actions)


        ratio_n = tf.exp(log_p_n - log_oldp_n)  #this to measure the divergence as in PPO loss
        Nf = tf.cast(N, dtype)


        surr = -tf.reduce_mean(ratio_n * advants) # Surrogate loss (non clipped loss)
       


        var_list = self.generator.trainable_weights



        kl = gauss_KL(oldaction_dist_mu, oldaction_dist_logstd,     #get old actuibs their std abd currebt actions and their std
                      action_dist_mu, action_dist_logstd) / Nf



        
        ent = gauss_ent(action_dist_mu, action_dist_logstd) / Nf   #calculate the entropy



        self.losses = [surr, kl, ent] #the are loss of the generaotro
        
        self.pg = flatgrad(surr, var_list)  #more of getting policy gradients flattern grads for the surrogate loss
 
        # KL divergence where first arg is fixed
        
        kl_firstfixed = gauss_selfKL_firstfixed(action_dist_mu,          #some fixinf in TRPO
                                                action_dist_logstd) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)  #getting the gradients of the fized kl with respect tp the car losr

        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)

 


        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape) #gett the size of the varaubles
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
       
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)] #some of the TRPO rule
        self.fvp = flatgrad(gvp, var_list)  #get a flat list

        self.gf = GetFlat(self.sess, var_list)  #this can evaluate the varuables

        self.sff = SetFromFlat(self.sess, var_list)  ##assigning varaibels vack to their orgianal shape ?? 
       
        
        self.baseline = NNBaseline(sess, feat_dim, aux_dim, encode_dim,   #This is to keep trackof updates same as the generator
                                   self.config.lr_baseline, self.config.b_iter,
                                   self.config.batch_size)

        
        self.sess.run(tf.global_variables_initializer())



        # Create feature extractor
        self.base_model = ResNet50(weights='imagenet', include_top=False)
        self.feat_extractor = Model(
            input=self.base_model.input,
            output=self.base_model.get_layer('activation_40').output
        )


    def create_generator(self, feats, auxs, encodes):  #This is where the generator get created
       
        feats = Input(tensor=feats) #get the resnet feature conv-4-f
        x = Convolution2D(256, 3, 3)(feats)
        x = LeakyReLU()(x)
        x = Convolution2D(256, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)


        auxs = Input(tensor=auxs)
        h = merge([x, auxs], mode='concat')
        h = Dense(256)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)

        encodes = Input(tensor=encodes)
        c = Dense(128)(encodes)

        h = merge([h, c], mode='sum')
        h = LeakyReLU()(h)



        #Next three seperatly filly connectead layers for each of the action output (steer,accel and beread)

        steer = Dense(1, activation='tanh', kernel_initializer='glorot_uniform')(h)


        accel = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(h)

        brake = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(h)

        
        actions = merge([steer, accel, brake], mode='concat')
        model = Model(input=[feats, auxs, encodes], output=actions)
        return model

    def create_discriminator(self, img_dim, aux_dim, action_dim):
        print("Bulding the discriminator")
        imgs = Input(shape=[img_dim[0], img_dim[1], img_dim[2]]) #small input iamge for the dicriminattor
        x = Convolution2D(32, 3, 3, subsample=(2, 2))(imgs)
        x = LeakyReLU()(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Convolution2D(128, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)

        auxs = Input(shape=[aux_dim]) #getting the auxilariy dims
        actions = Input(shape=[action_dim])

        h = merge([x, auxs, actions], mode='concat')  #addig up action and cx vector with the auxilary variables
        h = Dense(256)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        p = Dense(1)(h)
        discriminate = Model(input=[imgs, auxs, actions], output=p)  #settih up the discriminator as a Model to reuse

        imgs_n = Input(shape=[img_dim[0], img_dim[1], img_dim[2]]) 
        imgs_d = Input(shape=[img_dim[0], img_dim[1], img_dim[2]])
        auxs_n = Input(shape=[aux_dim])
        auxs_d = Input(shape=[aux_dim])
        actions_n = Input(shape=[action_dim])
        actions_d = Input(shape=[action_dim])

        p_n = discriminate([imgs_n, auxs_n, actions_n])  #agent
        p_d = discriminate([imgs_d, auxs_d, actions_d])  #expert
        p_d = Lambda(lambda x: -x)(p_d)  #pn-pd
        p_output = merge([p_n, p_d], mode='sum') #model output is the pn-pd  (can directly minimize this loss)


        model = Model(input=[imgs_n, auxs_n, actions_n,
                             imgs_d, auxs_d, actions_d],
                      output=p_output)      #model output is the pn-pd
        rmsprop = RMSprop(lr=self.config.lr_discriminator)
        model.compile(agent
            # little trick to use Keras predefined lambda loss function
            loss=lambda y_pred, p_true: K.mean(y_pred * p_true), optimizer=rmsprop   #if we provide a loss function with read and true this will get the mean
        )

        return model, discriminate #return the loss and the dicriminator

    def create_posterior(self, img_dim, aux_dim, action_dim, encode_dim):
        imgs = Input(shape=[img_dim[0], img_dim[1], img_dim[2]]) #get the normal image used to discriminator 

        x = Convolution2D(32, 3, 3, subsample=(2, 2))(imgs)
        x = LeakyReLU()(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Convolution2D(128, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)

        auxs = Input(shape=[aux_dim])
        actions = Input(shape=[action_dim])
        h = merge([x, auxs, actions], mode='concat') #concatoanate actions auximalry distributions and 
        h = Dense(256)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        c = Dense(encode_dim, activation='softmax')(h)

        model = Model(input=[imgs, auxs, actions], output=c)
        adam = Adam(lr=self.config.lr_posterior)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy'])      #use adam to train this using the supervise leearbg
        return model

    def act(self, feats, auxs, encodes, logstds, *args):
        
        action_dist_mu = \
                self.sess.run(
                    self.action_dist_mu,
                    {self.feats: feats, self.auxs: auxs, self.encodes: encodes}
                )




        act = action_dist_mu + np.exp(logstds) * \
                np.random.randn(*logstds.shape) #Adding some noise and deviation to the actions

        act[:, 0] = np.clip(act[:, 0], -1, 1)  #this is to clip the action values
        act[:, 1] = np.clip(act[:, 1], 0, 1)
        act[:, 2] = np.clip(act[:, 2], 0, 1)

        return act

    def learn(self, demo):
        config = self.config
        start_time = time.time()
        numeptotal = 0



        # Set up for training discrimiator
        print "Loading data ..."
        imgs_d, auxs_d, actions_d = demo["imgs"], demo["auxs"], demo["actions"]
        numdetotal = imgs_d.shape[0]
        idx_d = np.arange(numdetotal)
        np.random.shuffle(idx_d)




        print("___________________________________________________________________________________________________")

        

        imgs_d = imgs_d[idx_d]
        auxs_d = auxs_d[idx_d]
        actions_d = actions_d[idx_d]

        print "Resizing img for demo ..."  #original 110*200
        imgs_reshaped_d = []
        from PIL import Image



        #img2 = Image.fromarray(imgs_d[0], 'RGB')
        #img2.save('my2.png')
        #img2.show()
        #pdb.set_trace()

        for i in xrange(numdetotal):
            imgs_reshaped_d.append(cv2.resize(imgs_d[i],(self.img_dim[0], self.img_dim[1])))

        #imgs_d = np.concatenate(imgs_reshaped_d, axis=0).astype(np.float32)

        imgs_d = np.asarray(imgs_reshaped_d)



        

        #print(np.shape(imgs_d))
        #imgs_d = (imgs_d - 128.) / 128.
        print "Shape of resized demo images:", imgs_d.shape


  
        

        
        #img = Image.fromarray(imgs_reshaped_d[7999], 'RGB')
        #img.show()
        

        for i in xrange(38, config.n_iter):


            # Generating paths.
            # if i == 1:
            if i == 38:
                paths_per_collect = 30
            else:
                paths_per_collect = 10
            rollouts = rollout_contin(
                self.env,
                self,
                self.feat_extractor,
                self.feat_dim,
                self.aux_dim,
                self.encode_dim,
                config.max_step_limit,
                config.min_step_limit,
                config.pre_step,
                paths_per_collect,
                self.pre_actions,
                self.discriminate,
                self.posterior_target)

            

            for path in rollouts:
                self.buffer.add(path)
            print "Buffer count:", self.buffer.count()  #This also important keep a reply buffer


            paths = self.buffer.get_sample(config.sample_size)



            print "Calculating actions ..."
       
            for path in paths:  #for each path dictionary we add the  3 dimentional action probaliriws
                #print(path["actions"])
                #print("rrrrrrrrrrrrrrrrrrrrr_____________________rrrrrrrrrrrrrrrrr")
                path["mus"] = self.sess.run(  #this is to executing the 
                    self.action_dist_mu,                                     #again get the generator action output for each state and action part
                    {self.feats: path["feats"],
                     self.auxs: path["auxs"],
                     self.encodes: path["encodes"]}  #here they tale the actopm dostrobitopm without cliiping or adding deviation
                )
       
                

            mus_n = np.concatenate([path["mus"] for path in paths])  #32 separate action values this is the actions for each state without ading diviation or nosise
            logstds_n = np.concatenate([path["logstds"] for path in paths])
            feats_n = np.concatenate([path["feats"] for path in paths])
            auxs_n = np.concatenate([path["auxs"] for path in paths])
            encodes_n = np.concatenate([path["encodes"] for path in paths])
            actions_n = np.concatenate([path["actions"] for path in paths]) #these are the real action values used in rollouts
            imgs_n = np.concatenate([path["imgs"] for path in paths])    #32 images got while 
            imgs_l = [path["imgs"] for path in paths]

            from PIL import Image
            #img2 = Image.fromarray(imgs_l[0][0], 'RGB')
            #img2.show()
            #print(np.shape(imgs_l))
            #print(np.shape(imgs_l[0][0]))
            #pdb.set_trace()
            

            print "Epoch:", i, "Total sampled data points:", feats_n.shape[0]

            # Train discriminator
            numnototal = feats_n.shape[0]
            batch_size = config.batch_size
            start_d = self.demo_idx  #get the demos index like where shoid we start
            start_n = 0

            if i <= 5:
                d_iter = 120 - i * 20
            else:
                d_iter = 10

            for k in xrange(10):
                loss = self.discriminator.train_on_batch(
                    [imgs_n[start_n:start_n + batch_size], #from the agent
                     auxs_n[start_n:start_n + batch_size], #from the agent
                     actions_n[start_n:start_n + batch_size], #from the agent/orginal actions 
                     imgs_d[start_d:start_d + batch_size],  #have 3000 | 50*50*3 images 
                     auxs_d[start_d:start_d + batch_size],
                     actions_d[start_d:start_d + batch_size]],
                    np.ones(batch_size)
                )
                #print(loss)
                #print("zzzzzzzzzzzzzzzzzzzzzzzzzzzz")
                

                # print self.discriminator.summary()
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, config.clamp_lower, config.clamp_upper)
                               for w in weights]
                    l.set_weights(weights)

                start_d = self.demo_idx = self.demo_idx + batch_size
                start_n = start_n + batch_size

                if start_d + batch_size >= numdetotal:
                    start_d = self.demo_idx = (start_d + batch_size) % numdetotal
                if start_n + batch_size >= numnototal:
                    start_n = (start_n + batch_size) % numnototal

                print "Discriminator step:", k, "loss:", loss




            idx = np.arange(numnototal)
            np.random.shuffle(idx)
            train_val_ratio = 0.7
            # Training data for posterior
            numno_train = int(numnototal * train_val_ratio)
            imgs_train = imgs_n[idx][:numno_train]
            auxs_train = auxs_n[idx][:numno_train]
            actions_train = actions_n[idx][:numno_train]
            encodes_train = encodes_n[idx][:numno_train]  #we literaly know what isthe encoding we used to generate the tracks

            # Validation data for posterior  #validate how the posterior work 
            imgs_val = imgs_n[idx][numno_train:]
            auxs_val = auxs_n[idx][numno_train:]
            actions_val = actions_n[idx][numno_train:]
            encodes_val = encodes_n[idx][numno_train:]

            start_n = 0
            for j in xrange(config.p_iter):
                loss = self.posterior.train_on_batch(  #oginal posterior network this is actually the main network
                    [imgs_train[start_n:start_n + batch_size],
                     auxs_train[start_n:start_n + batch_size],
                     actions_train[start_n:start_n + batch_size]],
                    encodes_train[start_n:start_n + batch_size]
                )

                start_n += batch_size
                if start_n + batch_size >= numno_train:
                    start_n = (start_n + batch_size) % numno_train

                posterior_weights = self.posterior.get_weights()
                posterior_target_weights = self.posterior_target.get_weights()
                for k in xrange(len(posterior_weights)):  #after this we change the wiehts of posterior network   
                    posterior_target_weights[k] = 0.5 * posterior_weights[k] +\
                            0.5 * posterior_target_weights[k]
                self.posterior_target.set_weights(posterior_target_weights)

                output_p = self.posterior_target.predict(
                    [imgs_val, auxs_val, actions_val])
                val_loss = -np.average(
                    np.sum(np.log(output_p) * encodes_val, axis=1))
                print "Posterior step:", j, "loss:", loss, val_loss

            # Computing returns and estimating advantage function.

            path_idx = 0
       
            for path in paths:  #Again we take the rollout paths dictionory
                file_path = "/home/dl/Desktop/InfoGAIL-master/iter_%d_path_%d.txt" % (i, path_idx)
                f = open(file_path, "w")
                path["baselines"] = self.baseline.predict(path)  #add the baseline path to the paths
                
                
                output_d = self.discriminate.predict(  #get the probabilities from the discriminator
                    [path["imgs"], path["auxs"], path["actions"]])


                output_p = self.posterior_target.predict(  #get two score for the path  created by the  ge
                    [path["imgs"], path["auxs"], path["actions"]])

                
                path["rewards"] = np.ones(path["raws"].shape[0]) * 2 + \
                        output_d.flatten() * 0.1 + \
                        np.sum(np.log(output_p) * path["encodes"], axis=1)


                path_baselines = np.append(path["baselines"], 0 if   #stabalizing the gradient updates
                                           path["baselines"].shape[0] == 100 else
                                           path["baselines"][-1])


                deltas = path["rewards"] + config.gamma * path_baselines[1:] -\
                        path_baselines[:-1]

            
                # path["returns"] = discount(path["rewards"], config.gamma)
                # path["advants"] = path["returns"] - path["baselines"]
                path["advants"] = discount(deltas, config.gamma * config.lam) #discounted adcantage functions
                path["returns"] = discount(path["rewards"], config.gamma)

                f.write("Baseline:\n" + np.array_str(path_baselines) + "\n")
                f.write("Returns:\n" + np.array_str(path["returns"]) + "\n")
                f.write("Advants:\n" + np.array_str(path["advants"]) + "\n")
                f.write("Mus:\n" + np.array_str(path["mus"]) + "\n")
                f.write("Actions:\n" + np.array_str(path["actions"]) + "\n")
                f.write("Logstds:\n" + np.array_str(path["logstds"]) + "\n")
                path_idx += 1

            # Standardize the advantage function to have mean=0 and std=1
            advants_n = np.concatenate([path["advants"] for path in paths])
            # advants_n -= advants_n.mean()
            advants_n /= (advants_n.std() + 1e-8)



            # Computing baseline function for next iter.
            self.baseline.fit(paths) #training the baseline generator (These guys kept a differnet network)


            



            feed = {self.feats: feats_n,
                    self.auxs: auxs_n,
                    self.encodes: encodes_n,
                    self.actions: actions_n,
                    self.advants: advants_n,
                    self.action_dist_logstd: logstds_n,
                    self.oldaction_dist_mu: mus_n,
                    self.oldaction_dist_logstd: logstds_n}

            thprev = self.gf()



            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.sess.run(self.fvp, feed) + p * config.cg_damping

            g = self.sess.run(self.pg, feed_dict=feed)  #we get the policy gradients
            stepdir = conjugate_gradient(fisher_vector_product, -g)  #need this to computer TRPO 
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            assert shs > 0

            lm = np.sqrt(shs / config.max_kl)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)

            def loss(th):
                self.sff(th)
                return self.sess.run(self.losses[0], feed_dict=feed)
            theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
            self.sff(theta)

            surrafter, kloldnew, entropy = self.sess.run(
                self.losses, feed_dict=feed
            )

            episoderewards = np.array([path["rewards"].sum() for path in paths])
            stats = {}
            numeptotal += len(episoderewards)
            stats["Total number of episodes"] = numeptotal
            stats["Average sum of rewards per episode"] = episoderewards.mean()
            stats["Entropy"] = entropy
            stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
            stats["KL between old and new distribution"] = kloldnew
            stats["Surrogate loss"] = surrafter
            print("\n********** Iteration {} **********".format(i))
            for k, v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if entropy != entropy:
                exit(-1)

            param_dir = "/home/Desktop/InfoGAIL-master/params/"
            print("Now we save model")
            self.generator.save_weights(
                param_dir + "generator_model_%d.h5" % i, overwrite=True)
            with open(param_dir + "generator_model_%d.json" % i, "w") as outfile:
                json.dump(self.generator.to_json(), outfile)

            self.discriminator.save_weights(
                param_dir + "discriminator_model_%d.h5" % i, overwrite=True)
            with open(param_dir + "discriminator_model_%d.json" % i, "w") as outfile:
                json.dump(self.discriminator.to_json(), outfile)

            self.baseline.model.save_weights(
                param_dir + "baseline_model_%d.h5" % i, overwrite=True)
            with open(param_dir + "baseline_model_%d.json" % i, "w") as outfile:
                json.dump(self.baseline.model.to_json(), outfile)

            self.posterior.save_weights(
                param_dir + "posterior_model_%d.h5" % i, overwrite=True)
            with open(param_dir + "posterior_model_%d.json" % i, "w") as outfile:
                json.dump(self.posterior.to_json(), outfile)

            self.posterior_target.save_weights(
                param_dir + "posterior_target_model_%d.h5" % i, overwrite=True)
            with open(param_dir + "posterior_target_model_%d.json" % i, "w") as outfile:
                json.dump(self.posterior_target.to_json(), outfile)


'''

class Generator(object):
    def __init__(self, sess, feat_dim, aux_dim, encode_dim, action_dim):
        self.sess = sess
        self.lr = tf.placeholder(tf.float32, shape=[])

        K.set_session(sess)

        self.model, self.weights, self.feats, self.auxs, self.encodes = \
                self.create_generator(feat_dim, aux_dim, encode_dim)

        self.action_gradient = tf.placeholder(tf.float32, [None, action_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights,
                                        self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

        self.sess.run(tf.global_variables_initializer())

    def train(self, feats, auxs, encodes, action_grads, lr):
        self.sess.run(self.optimize, feed_dict={
            self.feats: feats,
            self.auxs: auxs,
            self.encodes: encodes,
            self.lr: lr,
            self.action_gradient: action_grads,
            K.learning_phase(): 1
        })

    def create_generator(self, feat_dim, aux_dim, encode_dim):
        feats = Input(shape=[feat_dim[0], feat_dim[1], feat_dim[2]])
        x = Convolution2D(256, 3, 3)(feats)
        x = LeakyReLU()(x)
        x = Convolution2D(256, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        auxs = Input(shape=[aux_dim])
        h = merge([x, auxs], mode='concat')
        h = Dense(256)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        encodes = Input(shape=[encode_dim])
        c = Dense(128)(encodes)
        h = merge([h, c], mode='sum')
        h = LeakyReLU()(h)

        steer = Dense(1, activation='tanh', init=lambda shape, name:
                      normal(shape, scale=1e-4, name=name))(h)
        accel = Dense(1, activation='sigmoid', init=lambda shape, name:
                             normal(shape, scale=1e-4, name=name))(h)
        brake = Dense(1, activation='sigmoid', init=lambda shape, name:
                      normal(shape, scale=1e-4, name=name))(h)
        actions = merge([steer, accel, brake], mode='concat')
        model = Model(input=[feats, auxs, encodes], output=actions)
        return model, model.trainable_weights, feats, auxs, encodes
'''

'''
class Posterior(object):
    def __init__(self, sess, img_dim, aux_dim, action_dim, encode_dim):
        self.sess = sess
        self.lr = tf.placeholder(tf.float32, shape=[])

        K.set_session(sess)

        self.model = self.create_posterior(img_dim, aux_dim, action_dim, encode_dim)

    def create_posterior(self, img_dim, aux_dim, action_dim, encode_dim):
        imgs = Input(shape=[img_dim[0], img_dim[1], img_dim[2]])
        x = Convolution2D(32, 3, 3, subsample=(2, 2))(imgs)
        x = LeakyReLU()(x)
        x = Convolution2D(64, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Convolution2D(128, 3, 3, subsample=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        auxs = Input(shape=[aux_dim])
        actions = Input(shape=[action_dim])
        h = merge([x, auxs, actions], mode='concat')
        h = Dense(256)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        c = Dense(encode_dim, activation='softmax')(h)

        model = Model(input=[imgs, auxs, actions], output=c)
        return model
'''
