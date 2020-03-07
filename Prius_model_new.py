# -*- coding: utf-8 -*-
"""
the Model of Prius
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d
import math
import scipy.io as scio

class Prius_model():
    def __init__(self):
        # paramsmeters of car
        self.Wheel_R = 0.287
        self.mass = 1449
        self.C_roll  = 0.013
        self.density_air = 1.2
        self.area_frontal = 2.23
        self.G = 9.81
        self.C_d = 0.26
        # the factor of F_roll
        self.T_factor = 1
       
        # paramsmeters of transmission system
        # number of teeth
        self.Pgs_R = 78
        self.Pgs_S = 30
        # speed ratio from ring gear to wheel
        self.Pgs_K = 3.93  
        
        # The optimal curve of engine   
        self.Eng_pwr_opt_list = np.arange(0, 57, 1) * 1000
        #T_list = [0.0, 11.93662073189215, 23.8732414637843, 35.80986219567645, 47.7464829275686, 59.68310365946075, 71.6197243913529, 78.64126599834829, 84.88263631567752, 76.73541899073525, 82.3215222889114, 82.71044286665428, 84.25849928394459, 89.30996806595566, 83.03736161316279, 87.87696244337779, 88.31719385446216, 92.7645954021333, 98.22133630814113, 98.6068669156308, 97.94150344116636, 99.768770296412, 101.98277906859313, 104.09185851507847, 96.70173757482249, 105.16846459816874, 101.75479968170357, 103.54658948147409, 105.683914780389, 107.75470855248945, 109.34309067382122, 110.04765581818786, 110.7164821508837, 112.14476417151343, 113.92143294998826, 116.05047933784036, 117.73105379400477, 119.36620731892151, 120.95775674984046, 121.70672118791997, 121.64709026132127, 121.96920872463008, 121.90591385762197, 121.84562408815725, 121.7881303659721, 121.73324259153468, 121.68078751624131, 121.96112486933283, 121.58255599593066, 121.85300330473238, 121.80225236624644, 121.75353146529994, 121.70672118791995, 121.36995660245255, 121.90591385762195, 121.85877313300573, 121.81335052135952]
        self.W_list = [91.106186954104, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 89.0117918517108, 94.24777960769379, 117.28612573401894, 121.47491593880532, 132.9940890019679, 142.4188669627373, 145.56045961632708, 168.59880574265222, 170.69320084504542, 181.1651763570114, 183.25957145940458, 183.25957145940458, 192.684349420174, 204.20352248333657, 210.48670779051614, 215.72269554649912, 220.95868330248211, 248.18581963359367, 237.71384412162766, 255.51620249196986, 260.7521902479528, 264.9409804527392, 269.1297706575256, 274.3657584135086, 281.69614127188476, 289.026524130261, 294.262511886244, 298.4513020910303, 301.59289474462014, 305.78168494940655, 309.9704751541929, 314.1592653589793, 320.4424506661589, 328.8200310757317, 336.15041393410786, 344.52799434368063, 352.90557475325346, 361.2831551628262, 369.660735572399, 378.0383159819718, 385.368698840348, 394.7934768011173, 402.12385965949346, 410.5014400690663, 418.87902047863906, 427.2566008882119, 436.6813788489813, 442.96456415616086, 451.3421445657336, 459.7197249753064]
        self.Eng_pwr_func = interp1d(self.Eng_pwr_opt_list, self.W_list)  
        
        Eng_spd_list = np.arange(0, 4501, 125) * (2 * math.pi) / 60
        Eng_spd_list = Eng_spd_list[np.newaxis, :]
        Eng_trq_list = np.arange(0, 111, 5) * (121 / 110)
        Eng_trq_list = Eng_trq_list[np.newaxis, :]
        data_path = 'Eng_bsfc_map.mat'
        data = scio.loadmat(data_path)
        Eng_bsfc_map = data['Eng_bsfc_map']    
        self.Eng_trq_maxP = [-4.1757e-009, 6.2173e-006, -3.4870e-003, 9.1743e-001, 2.0158e+001]
        Eng_fuel_map = Eng_bsfc_map * (Eng_spd_list.T * Eng_trq_list) / 3600 / 1000
        
        # fuel consumption (g)
        self.Eng_fuel_func = interp2d(Eng_trq_list, Eng_spd_list, Eng_fuel_map)
        
        # Motor 1
        # motor speed list (rad/s)
        Mot_spd_list = np.arange(-6000, 6001, 200) * (2 * math.pi) / 60        
        # motor torque list (Nm)
        Mot_trq_list = np.arange(-400, 401, 10)
        
        # motor efficiency map
        data_path1 = 'Mot_eta_quarter.mat'
        data1 = scio.loadmat(data_path1)
        Mot_eta_quarter = data1['Mot_eta_quarter']
        Mot_eta_alltrqs = np.concatenate(([np.fliplr(Mot_eta_quarter[:, 1:]), Mot_eta_quarter]), axis = 1)
        Mot_eta_map = np.concatenate(([np.flipud(Mot_eta_alltrqs[1:, :]), Mot_eta_alltrqs]))
        # motor efficiency
        self.Mot_eta_map_func = interp2d(Mot_trq_list, Mot_spd_list, Mot_eta_map)
        
        #  motor maximum torque
        Mot_trq_max_quarter = np.array([400,400,400,400,400,400,400,347.200000000000,297.800000000000,269.400000000000,241,221.800000000000,202.600000000000,186.400000000000,173.200000000000,160,148,136,126.200000000000,118.600000000000,111,105.800000000000,100.600000000000,96.2000000000000,92.6000000000000,89,87.4000000000000,85.8000000000000,83.2000000000000,79.6000000000000,76])
        Mot_trq_max_quarter = Mot_trq_max_quarter[np.newaxis, :]
        Mot_trq_max_list = np.concatenate((np.fliplr(Mot_trq_max_quarter[:, 1:]), Mot_trq_max_quarter), axis = 1)        
        # motor minimum torque 
        Mot_trq_min_list = - Mot_trq_max_list
        self.Mot_trq_min_func = interp1d(Mot_spd_list, Mot_trq_min_list, kind = 'linear', fill_value = 'extrapolate')
        self.Mot_trq_max_func = interp1d(Mot_spd_list, Mot_trq_max_list, kind = 'linear', fill_value = 'extrapolate')

        # Generator (Motor 2)
        # generator speed list (rad/s)
        Gen_spd_list = np.arange(-10e3, 11e3, 1e3) * (2 * math.pi) / 60
        Gen_trq_list = np.arange(-75, 76, 5) 
        
        # motor efficiency map
        Gen_eta_quarter = np.array([[0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000],[0.570000000000000,0.701190476190476,0.832380952380952,0.845500000000000,0.845500000000000,0.845500000000000,0.845500000000000,0.841181818181818,0.832545454545455,0.825204545454546,0.820886363636364,0.816136363636364,0.807500000000000,0.798863636363636,0.794113636363636,0.789795454545455],[0.570000000000000,0.710238095238095,0.850476190476190,0.872272727272727,0.880909090909091,0.883500000000000,0.883500000000000,0.879181818181818,0.870545454545455,0.864500000000000,0.864500000000000,0.863636363636364,0.855000000000000,0.846363636363636,0.841613636363636,0.837295454545455],[0.570000000000000,0.710238095238095,0.850476190476190,0.872272727272727,0.880909090909091,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.875727272727273,0.867090909090909],[0.570000000000000,0.710238095238095,0.850476190476190,0.876159090909091,0.889113636363636,0.896022727272727,0.900340909090909,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.894727272727273,0.886090909090909],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.898181818181818,0.889545454545454,0.886090909090909,0.894727272727273,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.710238095238095,0.850476190476190,0.872272727272727,0.880909090909091,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.886090909090909,0.894727272727273,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000]])
        Gen_eta_alltrqs = np.concatenate((Gen_eta_quarter[:, 1:], Gen_eta_quarter), axis = 1)    
        Gen_eta_map = np.concatenate(([np.flipud(Gen_eta_alltrqs[1:, :]), Gen_eta_alltrqs]))
        # efficiency of the electric generator
        self.Gen_eta_map_func = interp2d(Gen_trq_list, Gen_spd_list, Gen_eta_map)

        # generator maxmium torque
        Gen_trq_max_half = np.array([76.7000000000000,76.7000000000000,70.6160000000000,46.0160000000000,34.7320000000000,26.6400000000000,21.2000000000000,18.1160000000000,16.2800000000000,13.4000000000000,0])
        Gen_trq_max_half = Gen_trq_max_half[np.newaxis, :]
        Gen_trq_max_list = np.concatenate((np.fliplr(Gen_trq_max_half[:, 1:]), Gen_trq_max_half), axis = 1)
        # generator minimum torque
        Gen_trq_min_list = -Gen_trq_max_list
        self.Gen_trq_min_func = interp1d(Gen_spd_list, Gen_trq_min_list, kind = 'linear', fill_value = 'extrapolate')
        self.Gen_trq_max_func = interp1d(Gen_spd_list, Gen_trq_max_list, kind = 'linear', fill_value = 'extrapolate')

        # Battery
        # published capacity of one battery cell
        Batt_Q_cell = 6.5     
        # coulombs, battery package capacity
        self.Batt_Q = Batt_Q_cell * 3600     
        # resistance and OCV list
        Batt_rint_dis_list = [0.7,0.619244814,0.443380117,0.396994948,0.370210379,0.359869599,0.364414573,0.357095093,0.363394618,0.386654377,0.4] # ohm
        Batt_rint_chg_list = [0.7,0.623009741,0.477267027,0.404193372,0.37640518,0.391748667,0.365290105,0.375071555,0.382795632,0.371566564,0.36] # ohm
        Batt_vol_list  = [202,209.3825073,213.471405,216.2673035,218.9015961,220.4855042,221.616806,222.360199,224.2510986,227.8065948,237.293396] # V
        # resistance and OCV
        SOC_list = np.arange(0, 1.01, 0.1) 
        self.Batt_vol_func = interp1d(SOC_list, Batt_vol_list, kind = 'linear', fill_value = 'extrapolate')
        self.Batt_rint_dis_list_func = interp1d(SOC_list, Batt_rint_dis_list, kind = 'linear', fill_value = 'extrapolate')
        self.Batt_rint_chg_list_func = interp1d(SOC_list, Batt_rint_chg_list, kind = 'linear', fill_value = 'extrapolate')  

        #Battery current limitations
        self.Batt_I_max_dis = 196
        self.Batt_I_max_chg = 120 
        
        
    def run(self, car_spd, car_a, Eng_pwr_opt, SOC):   
        # Wheel speed (rad/s)
        Wheel_spd = car_spd / self.Wheel_R
    
        # Wheel torque (Nm) 地面作用于驱动轮的切向反作用力
        F_roll = self.mass * self.G * self.C_roll * (self.T_factor if car_spd > 0 else 0)
        F_drag = 0.5 * self.density_air * self.area_frontal * self.C_d *(car_spd ** 2)
        F_a = self.mass * car_a
        T = self.Wheel_R * (F_a + F_roll + F_drag )
        P_req = T * Wheel_spd
        
        # Engine  
        if Eng_pwr_opt > 56000:
            Eng_pwr_opt = 56000
        
        if (Eng_pwr_opt < 500) or (T < 0):    
            Eng_pwr_opt = 0  
            
        Eng_spd = self.Eng_pwr_func(Eng_pwr_opt)
        Eng_trq = Eng_pwr_opt / Eng_spd  
        
        # The minimum power of engine and braking energy recovery    
        if (Eng_pwr_opt < 500) or (T < 0):
            Eng_trq = 0
            Eng_spd = 0
            
        Eng_fuel_mdot = self.Eng_fuel_func(Eng_trq, Eng_spd)
        # maximum engine torque boundary (Nm)
        Eng_trq_max = np.polyval(self.Eng_trq_maxP, Eng_spd)
        # engine power consumption
        Eng_pwr = Eng_fuel_mdot * 42600
        inf_eng = (Eng_trq > Eng_trq_max)
        
        F_pgs = (Eng_trq / (self.Pgs_R + self.Pgs_S))   
        
        # motor rotating speed and torque
        Mot_spd = self.Pgs_K * Wheel_spd
        Mot_trq = T / self.Pgs_K - F_pgs * self.Pgs_R
        Mot_trq = (Mot_trq < 0) * (Mot_trq < self.Mot_trq_min_func(Mot_spd)) * self.Mot_trq_min_func(Mot_spd) +\
                  (Mot_trq < 0) * (Mot_trq > self.Mot_trq_min_func(Mot_spd)) * Mot_trq +\
                  (Mot_trq >= 0) * (Mot_trq > self.Mot_trq_max_func(Mot_spd)) * self.Mot_trq_max_func(Mot_spd) +\
                  (Mot_trq >= 0) * (Mot_trq < self.Mot_trq_max_func(Mot_spd)) * Mot_trq 
        
        Mot_trq = np.array(Mot_trq).flatten()          
        Mot_eta = (Mot_spd == 0) + (Mot_spd != 0) * self.Mot_eta_map_func(Mot_trq, Mot_spd * np.ones(1)) #need to edit        
        inf_mot = (np.isnan(Mot_eta)) + (Mot_trq < 0) * (Mot_trq < self.Mot_trq_min_func(Mot_spd)) + (Mot_trq >= 0) * (Mot_trq > self.Mot_trq_max_func(Mot_spd))
        Mot_eta[np.isnan(Mot_eta)] = 1        
        # Calculate electric power consumption
        Mot_pwr = (Mot_trq * Mot_spd <= 0) * Mot_spd * Mot_trq * Mot_eta + (Mot_trq * Mot_spd > 0) * Mot_spd * Mot_trq / Mot_eta
        
        # genertor rotating speed and torque 
        Gen_spd = (Eng_spd * (self.Pgs_R + self.Pgs_S) - Mot_spd * self.Pgs_R ) / self.Pgs_S
        Gen_trq = - F_pgs * self.Pgs_S
        Gen_eta = (Gen_spd == 0) + (Gen_spd != 0) * self.Gen_eta_map_func(Gen_trq, Gen_spd)
        inf_gen = (np.isnan(Gen_eta)) + (Gen_trq < 0) * (Gen_trq < self.Gen_trq_min_func(Gen_spd)) + (Gen_trq >= 0) * (Gen_trq > self.Gen_trq_max_func(Gen_spd))
        Gen_eta[np.isnan(Gen_eta)] = 1
        # Calculate electric power consumption
        Gen_pwr = (Gen_trq * Gen_spd <= 0) * Gen_spd * Gen_trq * Gen_eta + (Gen_trq * Gen_spd > 0) * Gen_spd * Gen_trq / Gen_eta
        
        Batt_vol = self.Batt_vol_func(SOC)
        Batt_pwr = Mot_pwr + Gen_pwr  
        Batt_rint = (Batt_pwr > 0) * self.Batt_rint_dis_list_func(SOC) + (Batt_pwr <= 0) * self.Batt_rint_chg_list_func(SOC)
        #columbic efficiency (0.9 when charging)
        Batt_eta = (Batt_pwr > 0) + (Batt_pwr <= 0) * 0.9
        Batt_I_max = (Batt_pwr > 0) * self.Batt_I_max_dis + (Batt_pwr <= 0) * self.Batt_I_max_chg
        
        # the limitation of Batt_pwr
        inf_batt_one = (Batt_vol ** 2 < 4 * Batt_rint * Batt_pwr)    
        if Batt_vol ** 2 < 4 * Batt_rint * Batt_pwr:
    #        Eng_pwr = Eng_pwr + Batt_pwr - Batt_vol ** 2 / (4 * Batt_rint)    
    #        Eng_trq = Eng_pwr / Eng_spd       
            Batt_pwr = Mot_pwr - Batt_vol ** 2 / (4 * Batt_rint)              # 放电功率过大以及充电功率过大？
            Batt_I = Batt_eta * Batt_vol / (2 * Batt_rint)
    #        print('battery power is out of bound')
        else:          
            Batt_I = Batt_eta * (Batt_vol - np.sqrt(Batt_vol ** 2 - 4 * Batt_rint * Batt_pwr)) / 0.8
               
        inf_batt = inf_batt_one + (np.abs(Batt_I) > Batt_I_max)
           
        # New battery state of charge
        SOC_new = - Batt_I / self.Batt_Q + SOC   
        # Set new state of charge to real values
        SOC_new = (np.conjugate(SOC_new) + SOC_new) / 2
        
        if SOC_new > 1:
            SOC_new = 1.0
        
        P_out = Eng_pwr_opt + Batt_pwr
        # Cost
        I = (inf_batt + inf_eng + inf_mot + inf_gen != 0)
        # Calculate cost matrix (fuel mass flow)
        cost = (Eng_pwr / 42600)
        
        out = {}
        out['P_req'] = P_req
        out['P_out'] = P_out
        out['Eng_spd'] = Eng_spd
        out['Eng_trq'] = Eng_trq
        out['Eng_pwr'] = Eng_pwr 
        out['Eng_pwr_opt'] = Eng_pwr_opt
        out['Mot_spd'] = Mot_spd
        out['Mot_trq'] = Mot_trq
        out['Mot_pwr'] = Mot_pwr  
        out['Gen_spd'] = Gen_spd
        out['Gen_trq'] = Gen_trq
        out['Gen_pwr'] = Gen_pwr
        out['SOC'] = SOC_new     
        out['Batt_vol'] = Batt_vol       
        out['Batt_pwr'] = Batt_pwr
        out['inf_batt'] = inf_batt
        out['inf_batt_one'] = inf_batt_one
        out['T'] = T
        out['Mot_eta'] = Mot_eta
        out['Gen_eta'] = Gen_eta
        
        return  out, cost, I

#Prius = Prius_model()
#out, cost, I = Prius.run(20, 1, 30000, 0.8)
