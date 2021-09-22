# -*- coding: utf-8 -*-
"""
DDPG_Prius
"""
#import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf
import numpy as np
from Prius_model_new import Prius_model
import scipy.io as scio
import matplotlib.pyplot as plt

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        
        self.sess.run(tf.global_variables_initializer())
                  

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]


    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net1, 100, activation=tf.nn.relu, name = 'l2', trainable=trainable)
            net3 = tf.layers.dense(net2, 50, activation=tf.nn.relu, name = 'l3', trainable=trainable)
            a = tf.layers.dense(net3, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')
        
            
    def loadmodel(self):
        self.saver = tf.train.Saver() 
        self.saver.restore(self.sess, 'Checkpoints/Prius/save_net.ckpt-' + str(i))

#####################  Testing Process  ####################

s_dim = 3
a_dim = 1
a_bound = 1
DDPG = DDPG(a_dim, s_dim, a_bound)
data_path = 'Data_Standard Driving Cycles/Standard_NEDC.mat'
data = scio.loadmat(data_path)
car_spd_one = data['speed_vector']
total_milage = np.sum(car_spd_one) / 1000
total_step = 0
step_episode = 0
mean_reward_all = 0
cost_Engine_list = []
cost_all_list = []
cost_Engine_100Km_list = []
mean_reward_list = []
list_even = []
list_odd = []
mean_discrepancy_list = []
SOC_final_list = []
Prius = Prius_model()

for i in range(500, 501):  # i represents the number value of checkpoints
    DDPG.loadmodel()
    SOC = 0.65
    SOC_origin = SOC
    ep_reward = 0
    ep_reward_all = 0
    step_episode += 1
    SOC_data = []
    P_req_list = []
    P_out_list = []
    Eng_spd_list = []
    Eng_trq_list = []
    Eng_pwr_list = []
    Eng_pwr_opt_list = []
    Gen_spd_list = []
    Gen_trq_list = []
    Gen_pwr_list = []
    Mot_spd_list = []
    Mot_trq_list = []
    Mot_pwr_list = []
    Batt_pwr_list = []
    inf_batt_list = []
    inf_batt_one_list = []
    Reward_list = []
    Reward_list_all = []
    T_list = []
    Mot_eta_list = []
    Gen_eta_list = []
    car_spd = car_spd_one[:, 0]
    car_a = car_spd_one[:, 0] - 0
    s = np.zeros(s_dim)
    s[0] = car_spd / 33.4
    s[1] = (car_a - (-1.5)) / (1.5- (-1.5))
    s[2] = SOC
    for j in range(car_spd_one.shape[1] - 1):
        a = DDPG.choose_action(s)   
        Eng_pwr_opt = (a[0]) * 56000
        
        out, cost, I = Prius.run(car_spd, car_a, Eng_pwr_opt, SOC)
        P_req_list.append(float(out['P_req']))
        P_out_list.append(float(out['P_out']))
        Eng_spd_list.append(float(out['Eng_spd']))
        Eng_trq_list.append(float(out['Eng_trq'])) 
        Eng_pwr_list.append(float(out['Eng_pwr']))
        Eng_pwr_opt_list.append(float(out['Eng_pwr_opt']))
        Mot_spd_list.append(float(out['Mot_spd']))
        Mot_trq_list.append(float(out['Mot_trq']))        
        Mot_pwr_list.append(float(out['Mot_pwr']))  
        Gen_spd_list.append(float(out['Gen_spd']))
        Gen_trq_list.append(float(out['Gen_trq']))        
        Gen_pwr_list.append(float(out['Gen_pwr']))
        Batt_pwr_list.append(float(out['Batt_pwr']))   
        inf_batt_list.append(int(out['inf_batt']))
        inf_batt_one_list.append(int(out['inf_batt_one']))
        Mot_eta_list.append(float(out['Mot_eta']))
        Gen_eta_list.append(float(out['Gen_eta']))
        T_list.append(float(out['T'])) 
        SOC_new = float(out['SOC'])
        SOC_data.append(SOC_new)
        cost = float(cost)
        r = cost
        ep_reward += r
        Reward_list.append(r)
        
        if SOC_new < 0.6 or SOC_new > 0.85:
            r = ((350 * ((0.6 - SOC_new) ** 2)) + cost)
        
        # Obtained from the wheel speed sensor            
        car_spd = car_spd_one[:, j + 1]
        car_a = car_spd_one[:, j + 1] - car_spd_one[:, j]
        s_ = np.zeros(s_dim)
        s_[0] = car_spd / 33.4   
        s_[1] = (car_a - (-1.5)) / (1.5- (-1.5))
        s_[2] = SOC_new
        
        s = s_
        ep_reward_all += r
        Reward_list_all.append(r)
        total_step += 1
        SOC = SOC_new
        cost_Engine = (ep_reward / 0.72 / 1000)
        cost_all = (ep_reward_all / 0.72 / 1000)

        if j == (car_spd_one.shape[1] - 2):
            SOC_final_list.append(SOC)
            mean_reward = ep_reward_all / car_spd_one.shape[1]
            mean_reward_list.append(mean_reward)
            cost_Engine += (SOC < SOC_origin) * (SOC_origin - SOC) * (201.6 * 6.5) * 3600 /(42600000) / 0.72 
            cost_Engine_list.append(cost_Engine)
            cost_Engine_100Km_list.append(cost_Engine * (100 / total_milage))
            cost_all += (SOC < SOC_origin) * (SOC_origin - SOC) * (201.6 * 6.5) * 3600 /(42600000) / 0.72 
            cost_all_list.append(cost_all)
            print('Episode:', i, ' cost_Engine: %.3f' % cost_Engine, ' Fuel_100Km: %.3f' % (cost_Engine * (100 / total_milage)), ' SOC-final: %.3f' % SOC)          
   
    mean_reward_all += mean_reward   
    if (step_episode % 10) == 0 and step_episode >= 10:
        if (step_episode / 10) % 2 == 0:
            list_even.append(mean_reward_all)
        else:
            list_odd.append(mean_reward_all)
        mean_reward_all = 0 
    
mean_discrepancy_list = list(map(lambda x, y: y - x, list_even, list_odd))   
x = np.arange(0, len(SOC_data), 1)
y = SOC_data
plt.plot(x, y)
plt.xlabel('time')
plt.ylabel('SOC')