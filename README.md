# DRL-based-EMS

A rule-interposing deep reinforcement learning (RIDRL) based energy management strategy (EMS) of hybrid electric vehicle (HEV) is investigated. Incorporated with the battery characteristics and the optimal brake specific fuel consumption (BSFC) curve of hybrid electric vehicles (HEVs), we are committed to solving the optimization problem of multi-objective energy management with a large space of control variables.
 
 ## Prius modelling
As shown in Fig. 1, the core power-split component of Prius is a planetary gear (PG) which splits power among the engine, motor and generator. In this structure, its engine and generator are connected with the planet carrier and sun gear respectively, and its motor is connected with the ring gear that is linked with the output shaft simultaneously. In addition, Prius is equipped with a small capacity Nickel metal hydride (Ni-MH) battery which is used to drive the traction motor and generator. Prius combines the advantages of series and parallel HEVs, and consists of three driving modes: pure electric mode, hybrid mode and charging mode.
 
In this research, a backward HEV model is built for the training and evaluation of EMS [32]. The vehicle power demand under the given driving cycle is calculated by the longitudinal force balance equation. The engine, generator and motor are modeled by their corresponding efficiency maps from bench experiments. The Ni-MH battery is modeled by an equivalent circuit model, wherein the impact of the temperature change and battery aging are not considered. 

 <div align="center"><img width="350" src="https://github.com/lryz0612/Image/blob/master/Prius.jpg"/><img width="450" src="https://github.com/lryz0612/Image/blob/master/engine%20map%20%26%20battery.jpg"/></div>
 
 ## DRL-based energy management strategy

<div align="center"><img height="350" src="https://github.com/lryz0612/Image/blob/master/DRL.jpg"/></div>
<center>Fig.3.Agent-environment interaction for HEV energy management</center>

<center>如何居中</center>

State = {SoC, velocity, acceleration}

Action = {continuous action: engine power}

### Evaluation metrics of DRL-based EMS
 
 ## Simulation results
 The simplified action space improves the convergence efficiency by 70.6%. The learning efficiency and performance robustness of RI DDPG make it more propitious to real applications in HEVs.
 
 
