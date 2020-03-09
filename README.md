# Deep reinforcement learning based energy management strategy for hybrid electric vehicle
**This research is cited from: Lian R, Peng J, Wu Y, et al. Rule-interposing deep reinforcement learning based energy management strategy for power-split hybrid electric vehicle. Energy, 2020: 117297.**

A rule-interposing deep reinforcement learning (RIDRL) based energy management strategy (EMS) of hybrid electric vehicle (HEV) is investigated. Incorporated with the battery characteristics and the optimal brake specific fuel consumption (BSFC) curve of hybrid electric vehicles (HEVs), we are committed to solving the optimization problem of multi-objective energy management with a large space of control variables.
 
 ## Prius modelling
As shown in Fig. 1, the core power-split component of Prius is a planetary gear (PG) which splits power among the engine, motor and generator. In this structure, its engine and generator are connected with the planet carrier and sun gear respectively, and its motor is connected with the ring gear that is linked with the output shaft simultaneously. In addition, Prius is equipped with a small capacity Nickel metal hydride (Ni-MH) battery which is used to drive the traction motor and generator. Prius combines the advantages of series and parallel HEVs, and consists of three driving modes: pure electric mode, hybrid mode and charging mode.
 
In this research, a backward HEV model is built for the training and evaluation of EMS. The vehicle power demand under the given driving cycle is calculated by the longitudinal force balance equation. The engine (Fig. 2), generator and motor are modeled by their corresponding efficiency maps from bench experiments. The Ni-MH battery is modeled by an equivalent circuit model, wherein the impact of the temperature change and battery aging are not considered. The experiment data of battery including internal resistance and open-circuit voltage is shown in Fig. 2.

 <div align="center"><img width="350" src="https://github.com/lryz0612/Image/blob/master/Prius.jpg"/><img width="450" src="https://github.com/lryz0612/Image/blob/master/engine%20map%20%26%20battery.jpg"/></div>
&emsp;&emsp;&emsp; Fig. 1. Architecture of Prius powertrain &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 2. Engine map and battery characteristics
 
 ## DRL-based energy management strategy
DRL agent is encountered with an environment with Markov property. The agent and the environment interact continually, the agent selecting actions and the environment responding rewards to these actions and presenting new states to the agent. In this research, deep deterministic policy gradient (DDPG) algorithm is incorporated with the expert knowledge of HEV to learn the optimal EMS action. Fig. 3 shows the agent-environment interaction of HEV energy management, i.e. the interaction between the EMS and the vehicle and traffic information. The state and action variables are set as follows, where the continuous action variables are explored from the optimal BSFC curve of engine. The reward function of DDPG-based EMS consists of two parts: the instantaneous fuel consumption of engine and the cost of battery charge sustaining. Thus, the multi-objective reward function is defined as:

**State = {SoC, velocity, acceleration}**

**Action = {continuous action: engine power}**

**$Reward = -\{\alpha[fuel(t)]+ \beta[SoC_{ref} - SoC(t)]^{2}\}$**

where α is the weight of fuel consumption, β the weight of battery charge sustaining, and $SoC_{ref}$ the SoC reference value while maintaining battery charge-sustaining.

<div align="center"><img height="350" src="https://github.com/lryz0612/Image/blob/master/DRL.jpg"/></div>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 3. Agent-environment interaction for HEV energy management

 ## Simulation results
We make extensive comparison experiments between RI DDPG and RI DQL, wherein they share the same embedded expert knowledge. The performance of the two models are evaluated according to the evaluation metrics defined in section 3.3.
Fig. 4 shows the differences of SoC trajectories among DP, RI DDPG and RI DQL under new European driving cycle (NEDC), where their values of terminal SoC are much the same, approximately at 0.6. From Fig. 5 and Fig. 6, it can be found that most of the engine working points of DDPG are distributed in areas with lower equivalent fuel consumption rate, while those of RI DQL are relatively poor. For this reason, the fuel economy of RI DDPG reaches 95.3% of DP’s, and get a decrease of 6.5% compared to that of RI DQL, as shown in table 5. In Fig. 7, it can be seen that RI DQL is difficult to guarantee its convergence and fluctuates more frequently as compared with RI DDPG that converges to a stable state after 50th episode. In order to train an EMS for a HEV online, the training process of a controller must be stable enough to guarantee the safety of powertrain. The stability of RI DDPG shows that it is more applicable to real-world applications of DRL-based EMSs.
For further verification, different driving cycles are introduced into the two EMSs. The simulation results in table 1 demonstrate the superiority of RI DDPG algorithm in performance robustness, where the mean and standard deviation of fuel economy are improved by 8.94% and 2.74% respectively.


<div align="center"><img height="250" src="https://github.com/lryz0612/Image/blob/master/SoC%20trajectories%20of%20the%20three%20EMS%20models.jpg"/></div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 4. SoC trajectories of the three EMS models

<div align="center"><img height="250" src="https://github.com/lryz0612/Image/blob/master/Working%20points%20of%20engine.jpg"/></div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 5. Working points of engine

<div align="center"><img width="420" src="https://github.com/lryz0612/Image/blob/master/brake%20specific%20fuel%20consumption.jpg"/><img width="420" src="https://github.com/lryz0612/Image/blob/master/Convergence%20curves.jpg"/></div>
&emsp;&emsp;&emsp; Fig. 6. Distributions of fuel consumption rate &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 7. Convergence curves 

<div align="center"><img height="250" src="https://github.com/lryz0612/Image/blob/master/Comparison%20between%20RI%20DDPG%20and%20RI%20DQL.jpg"/></div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 



