import random
import datetime
import itertools
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import simpy
from buffer import EpBuffer
import logging
from collections import deque

sys.setrecursionlimit(5000)

#create a looger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler("./logs/PPO_system.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

loggerTwin = logging.getLogger(__name__ + "Twin")
loggerTwin.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler("./logs/Twin_system.log")
file_handler.setFormatter(formatter)
loggerTwin.addHandler(file_handler)


logger_state = logging.getLogger(__name__ + "State_Evolution")
logger_state.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler("./logs/State_Evolution.log")
file_handler.setFormatter(formatter)
logger_state.addHandler(file_handler)
PLOT = False
number_machines = 2
# MU_Process = 1
MU_Sche = 1 
SI_Sche = 20








class Buffers():
    """Creates buffers"""

    def __init__(self, env, name):
        self.env = env # simpy object that controls the discrete event simulation
        self.name = name
        self.store = simpy.Store(env)  # Infinite Capacity


class Machine:
    """ Creates resources """

    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.isResource = simpy.Resource(env, capacity=1)


class ProductionSystem:

    def __init__(self, env, processing_time_range, initial_wip_cap, decision_epoch_interval, track_state_interval, warmup_time, policy, ep_buffer, random_seeds=(1, 2, 4), files=False, use_seeds=False, twin_system=None, logging = False):
        if not logging:
            logger.disabled = True
            loggerTwin.disabled = True
            logger_state.disabled = True
        self.warmup_time = warmup_time
        self.policy = policy
        self.ep_buffer = ep_buffer
        self.random_seeds = random_seeds
        self.use_seeds = use_seeds
        self.sum_rewards = 0
        self.average_wip = 1
        self.twin_system = twin_system
        self.track_state_interval = track_state_interval
        self.decision_epoch_interval = decision_epoch_interval

        self.previous_exit_time = 0

        self.weighted_average_beta = 0.9
        # State --> average Overal WIP; average interarrival time, std interarrival time; u_machine 1, u_machine 2, average_processing_time1, average_processing_time2
        # used in update functions and in decision epoch
        self.state = [0, 0, 0, 0, 0, 0, 0]
        # Tracks how many times an state element is updated during an epoch
        self.state_element_number_updates = [0, 0, 0, 0, 0, 0, 0]
        self.files = files
        self.env = env
        self.name = "Conwip" + " var = " + \
            str(processing_time_range) + " WIPCap = " + str(initial_wip_cap)
        self.processing_time_range = processing_time_range
        self.wip_cap = initial_wip_cap
        self.machines = []
        self.buffers = []
        self.order_buffer = None
        self.parts_produced = 0

        self.create_stations = self.stations()

        # creates a process that controls epochs end
        self.env.process(self.decision_epoch_action_input_process())
        # creates a process that triggers state elements update
        self.env.process(self.cyclical_state_elements_process())

        # Creates the process that will change WIP cap according to a schedule

        self.create_part_process = self.env.process(self.create_parts())

        self.r1 = np.random.RandomState(self.random_seeds[0])
        self.r2 = np.random.RandomState(self.random_seeds[1])
        self.r_schedule = np.random.RandomState(self.random_seeds[2])
        self.action_selection_RandomState = np.random.RandomState(
            self.random_seeds[0])

        # create directory to store files
        if self.files:
            self.path = "./" + datetime.datetime.now().strftime("%Y_%m_%d %H_%M_%S") +\
                        " var = " + str(processing_time_range) + " " + policy.name
            os.mkdir(self.path)

            self.tracking_start = self.env.process(self.track_wip_and_parts())

    def stations(self):

        self.order_buffer = Buffers(self.env, "Order Buffer")
        for i in range(number_machines):
            self.machines.append(Machine(self.env, "Machine {}".format(i)))
            self.buffers.append(Buffers(self.env, "Buffer {}".format(i)))

    def create_parts(self):

        count = 1
        while True:

            # timeout = self.r_schedule.lognormal(MU_Sche, SI_Sche)
            Part("Part {}".format(count), self)
            count += 1
            if self.use_seeds:
                timeout = max(0, self.r_schedule.normal(MU_Sche, SI_Sche))
            else:
                timeout = max(0, np.random.normal(MU_Sche, SI_Sche))

            # with open(self.name + " Sched" + ".csv", "a") as f:
            #     f.write(str(timeout) + "\n")

            yield self.env.timeout(timeout)

            self.update_average_interarrival_time_state_element(timeout)

    def compute_wip_level(self):
        wip = 0
        #counts all the parts waiting in the queue for the machines and the ones that
        #are being processed at the machines
        for machine in self.machines:
            wip += machine.isResource.count + len(machine.isResource.queue)
        return wip

    def compute_utilization(self):
        """Returns the number of parts at machines (m1, m2) """
        u_m1 = self.machines[0].isResource.count
        u_m2 = self.machines[1]. isResource.count

        return (u_m1, u_m2)

    def track_wip_and_parts(self):
        """Writes 2 files WIP.cvs and Parts Produced.cvs, the first tracks the WIP along the simulation 
           the second tracks the number o parts that already exited the system along the simulation
        """

        while True:
            yield self.env.timeout(0.5)
            with open(self.path + "/" + " WIP.csv", 'a') as f:
                f.write(str(self.env.now) + ";" +
                        str(self.compute_wip_level()) + ";" + str(self.wip_cap) + "\n")
            with open(self.path + "/" + " Parts Produced.csv", 'a') as f:
                f.write(str(self.env.now) + ";" +
                        str(self.parts_produced) + "\n")

    def track_flow_time(self, env, part):
        """Generates file (flow_time.csv) that will contain information on where is the part waiting"""

        with open(self.path + "/" + " flow_time" + ".csv", 'a') as f:
            # On order burffer, on shop floor, Total
            f.write(str(part.name) + str(';') + str(part.inf_e-part.inf_s) +
                    str(';') + str(part.shop_e - part.shop_s) +
                    str(";") + str(part.shop_e - part.inf_s) + "\n")

    def decision_epoch_action_input_process(self):
        """Defines the process where the WIP levels allowed will be changed
           And where state transitions will be store in a replay buffer"""
        trigger = True
        while True:
            # if warmup is finished we will start using our policy to take action instead of a random action
            
            if self.env.now > self.warmup_time:
                if trigger:
                    logger_state.debug(f"Warmup Time has ended")
                    trigger = False
                    
                # For state elements that need an event to be updated, like processing time and interarrival time, it may happen that
                # during the time in between decision making, there was no event tha led to update to that variable, this will cause an error
                #because we use an exponential moving avg with bias correction, no updates means division by zero, 
                #also it would mean avg processing time and interarrival time > than decision making interval
                for i in range(len(self.state)):

                    if self.state_element_number_updates[i] == 0:
                        self.state_element_number_updates[i] = 1
                        self.state[i] = self.decision_epoch_interval

                self.previous_state = np.array(
                    self.state) / (1 - self.weighted_average_beta ** np.array(self.state_element_number_updates))
                
                self.previous_state = np.reshape(self.previous_state, (1, -1))
                # print(self.previous_state)
                # We should pass an array to action, so that we get Wip as an integer
                self.action = self.policy.get_action(self.previous_state) #int

                self.reward = self.compute_reward(  
                    self.previous_state, self.action) #int

                self.wip_cap = self.action # int

                self.state = [0, 0, 0, 0, 0, 0, 0]
                self.state_element_number_updates = [0, 0, 0, 0, 0, 0, 0]

                yield self.env.timeout(self.decision_epoch_interval)
                # <<<<
                for i in range(len(self.state)):

                    if self.state_element_number_updates[i] == 0:
                        self.state_element_number_updates[i] = 1
                        self.state[i] = self.decision_epoch_interval

                state = np.array(self.state) / (1 - self.weighted_average_beta **
                                                     np.array(self.state_element_number_updates))
                
                
                # self.reward = self.compute_reward(state, self.action)
                if self.twin_system != None:

                    sample = (self.previous_state.reshape(self.previous_state.shape[1],), 
                        np.array(self.action), state,    self.reward)
                    # sample = list(itertools.chain(*sample))
                    self.ep_buffer.add_transition(sample)

            else:
                self.wip_cap = self.action_selection_RandomState.randint(1, 100)
                yield self.env.timeout(self.decision_epoch_interval)

    def cyclical_state_elements_process(self):
        """
        Orders the update of variables that need to be tracked cyclically, for variables like interarrival time there is an 
        event that triggers the update, in this case the arrival of another part. For variables like Avg utilization and Average WIP, there is no
        even to trigger its update, therefore we have to create a process to update those 2 variables
        """
        while True:
            yield self.env.timeout(self.track_state_interval)
            
            self.update_average_wip_state_element()
            self.update_utilization()

    def update_average_wip_state_element(self):
        # State --> average Overal WIP; average interarrival time, std interarrival time; u_machine 1, u_machine 2, average_processing_time1, average_processing_time2
        #track the number of times the this element of the sate was updated
        self.state_element_number_updates[0] += 1

        #Computes the exponential moving average of WIP
        #Bias correction will also be implemented in decision_epoch_action_input_process method
        self.state[0] = (self.state[0] * self.weighted_average_beta +
                         self.compute_wip_level() * (1 - self.weighted_average_beta))

        self.average_wip = self.average_wip * self.weighted_average_beta + \
            self.compute_wip_level() * (1 - self.weighted_average_beta)

    def update_flow_time_state_elements(self, flow_time):
        ###NOT IN USE <------------------------
        """inputs: flow time --> expects integer or float """
        self.state_element_number_updates[1] += 1
        self.state_element_number_updates[2] += 1

        # updates flow time
        self.state[1] = (self.state[1] * self.weighted_average_beta +
                         flow_time * (1 - self.weighted_average_beta))

        # update flow time std
        self.state[2] = (self.state[2] * self.weighted_average_beta +
                         (flow_time - self.state[1]) * (1 - self.weighted_average_beta))

    def update_average_interarrival_time_state_element(self, inter_arrival_time):
        """inputs: flow time --> expects integer or float """

        self.state_element_number_updates[1] += 1
        self.state_element_number_updates[2] += 1
        # updates flow time
        self.state[1] = (self.state[1] * self.weighted_average_beta +
                         inter_arrival_time * (1 - self.weighted_average_beta))
        self.state[2] = (self.state[2] * self.weighted_average_beta +
                         (inter_arrival_time - self.state[3]) * (1 - self.weighted_average_beta))

    def update_throughput_time(self, throughput_time):
        # State --> average Overal WIP; average interarrival time, std interarrival time; u_machine 1, u_machine 2, average_processing_time1, average_processing_time2
        ###NOT IN USE <------------------------
        
        self.state_element_number_updates[5] += 1
        self.state_element_number_updates[6] += 1

        #Compute average Processing time for machine 1
        self.state[5] = self.state[5] * self.weighted_average_beta + \
            throughput_time * (1 - self.weighted_average_beta)
        #Compute average Processing time for machine 1
        self.state[6] = self.state[6] * self.weighted_average_beta + \
            (throughput_time - self.state[5]) * \
            (1 - self.weighted_average_beta)

    def update_utilization(self):

        u_1, u_2 = self.compute_utilization()

        u_1 = max(1e-8, u_1) # in case utilization happens to be zero 
        u_2 = max(1e-8, u_2) 
        
        self.state_element_number_updates[3] += 1
        self.state_element_number_updates[4] += 1

        self.state[3] = self.state[3] * self.weighted_average_beta + \
            u_1 * (1 - self.weighted_average_beta)

        self.state[4] = self.state[4] * self.weighted_average_beta + \
            u_2 * (1 - self.weighted_average_beta)

    def update_processing_time(self, processing_time, machine_index):
        if machine_index == 0:
            self.state_element_number_updates[5] += 1
            self.state[5] = self.state[5] * self.weighted_average_beta + \
                processing_time * (1 - self.weighted_average_beta)

        if machine_index == 1:
            self.state_element_number_updates[6] += 1
            self.state[6] = self.state[6] * self.weighted_average_beta + \
                processing_time * (1 - self.weighted_average_beta)

    def compute_reward(self, state,  action):
        
        # throughput = (1 / state[0, 5])
        # flow_time = state[0, 1]
        # bottleneck_cycle_time = (self.processing_time_range[1] - self.processing_time_range[0]) / 2
        # raw_processing_time = (bottleneck_cycle_time) * 2
        # Check for the bottleneck station
        try:
            bottleneck_station = np.argmax(np.array([state[0, 5], state[0, 6]]))
        except RuntimeWarning:
            print("processing time machine 1 ", state[0, 5])
            print("processing time machine 2 ", state[0, 6])
        # Find the utilization of the bottleneck
        if bottleneck_station == 0:
            u = state[0, 3]
        elif bottleneck_station == 1:
            u = state[0, 4]

        # reward = ((throughput * bottleneck_cycle_time *
        #            flow_time) / raw_processing_time) + u
        # Only compute competition based reward signal for RL Agent    
        if self.twin_system != None:
            logger_state.debug(f"State: {state}")
            logger_state.debug(f"Twin state : {self.twin_system.previous_state}")           
            # Check for the bottlenec of the twin system
            bottleneck_station_twin = np.argmax(np.array([self.twin_system.previous_state[0, 5], self.twin_system.previous_state[0, 6]]))
            # Find the utilization of the bottleneck of the twin system
            if bottleneck_station_twin == 0:
                u_twin = self.twin_system.previous_state[0, 3] 
            if bottleneck_station_twin == 1:
                u_twin = self.twin_system.previous_state[0, 4]
            #if Utilization is higher than utilization of the twin and WIP is lower (best case scenario)
            
            logger_state.debug(f"U system: {u}")
            logger_state.debug(f"U twin system: {u_twin}")

            
            
            if u >= u_twin and state[0, 0] < self.twin_system.previous_state[0, 0]:
                # Reward is proportional to how better u is, and how much lower wip is
                reward = (u - u_twin) + \
                    (self.twin_system.previous_state[0, 0] - state[0, 0])
            #if Utilization is higher but wip is also higher
            elif u >= u_twin:
                #Reward porpotional to the the system that was the better ratio u/wip
                reward = (u/(state[0, 0] + 1e-8)) - (u_twin/(self.twin_system.previous_state[0, 0] + 1e-8))
          
            else:
                #if utilization is lower and wip also lower
                if u < u_twin and state[0, 0] < self.twin_system.previous_state[0, 0]:
                    #Reward porpotional to the the system that was the better ratio u/wip
                    reward = (u/(state[0, 0] + 1e-8)) - (u_twin/(self.twin_system.previous_state[0, 0] + 1e-8))

                #if utilization is lower and wip is higher (worst case scenario)
                elif u < u_twin and state[0, 0] >= self.twin_system.previous_state[0, 0]:
                    # the greater the difference in utilization the more negative will the reward be
                    # the greater the difference in wip between twin and state the more negative will the reward be
                    reward = (u - u_twin) + \
                        (self.twin_system.previous_state[0, 0] - state[0, 0])

            # wip_twin = self.twin_system.state[0, 0]
            # reward = u - (self.twin_system.previous_state[0, 0] - state[0, 0])
            # reward = reward + (self.parts_produced - self.twin_system.parts_produced)
        
        else:

            reward = u - state[0, 0] # dummy reward not to be used anywhere for the twin system

           
            
        self.sum_rewards = self.sum_rewards + reward  # track the sum of rewards
        return reward
        # return (((throughput*1000)/flow_time) - abs(self.wip_cap - action))


class Part():

    def __init__(self, name, production_system):
        self.env = production_system.env
        self.production_system = production_system
        self.name = name
        self.machines = production_system.machines
        self.buffers = production_system.buffers
        self.order_buffer = production_system.order_buffer
        # time it starts as information
        self.inf_s = None
        # time it exits information buffer (pre-shopfloor)
        self.inf_e = None
        # time it starts as a part in shop-floor (shop-floor release)
        self.shop_s = None
        # time it exits shop-floor (shop-floor exit)
        self.shop_e = None
        self.done = False
        self.processing_time_range = production_system.processing_time_range
        # start processing
        self.env.process(self.processing())

    def processing(self):
        self.order_buffer.store.put(self)

        self.inf_s = self.env.now           # Entered plant as information
        # print("Entrou ", self.name, " ", self.inf_s)
        if self.production_system.twin_system == None:
            loggerTwin.debug(f"{self.name} entered the system as information at {self.inf_s}")
        else:
            logger.debug(f"{self.name} entered the system as information at {self.inf_s}")
            


        # mantains env cycling until condition is met otherwise the simpy will stop cycling
        # yield self.env.event()

        while not self.done:
            yield self.env.timeout(0)
            # If wip level is not at the limited allowed and the part being called is the part that is waiting on queue longest or there is no part on the queue
            # Let the part enter the production system
            if self.production_system.compute_wip_level() < self.production_system.wip_cap and (self.production_system.order_buffer.store.items[0].name == self.name or len(self.production_system.order_buffer.store.items) == 0):
                self.order_buffer.store.get()
                # Order stoped being just information
                if self.production_system.twin_system == None:
                    loggerTwin.debug(f"{self.name}, released into production at {self.env.now}")

                else: 
                    logger.debug(f"{self.name}, released into production at {self.env.now}")
                    
                self.buffers[0].store.put(self)
                
                if self.production_system.twin_system == None:
                    loggerTwin.debug(f"{self.name}, entered Buffer 1 at  {self.env.now}")
               
                else: 
                    logger.debug(f"{self.name}, entered Buffer 1 at  {self.env.now}")
                

                self.inf_e = self.env.now    # Buffer 1
                self.shop_s = self.env.now        # Order released into shop floor

                with self.machines[0].isResource.request() as request:  # Machine 1

                    yield request

                    self.buffers[0].store.get()
                    
                    if self.production_system.twin_system == None:
                        loggerTwin.debug(f"{self.name}, started being processed on Machine1 at  {self.env.now}")
               
                    else: 
                        logger.debug(f"{self.name}, started being processed on Machine1 at  {self.env.now}")
                

                    # yield self.env.timeout(generate_random_machines(MU_Process, self.si_process, self.production_system))
                    if self.production_system.use_seeds:
                        time1 = self.production_system.r1.uniform(
                            self.processing_time_range[0], self.processing_time_range[1])
                    else:
                        time1 = np.random.uniform(
                            self.processing_time_range[0], self.processing_time_range[1])
                    # Update processing time average for m2
                    self.production_system.update_processing_time(time1, 0)

                    # time1 = self.production_system.r1.lognormal(
                    #     MU_Process, self.si_process)

                    # with open(self.production_system.name + " M1" + ".csv", "a") as f:
                    #     f.write(str(time1) + "\n")

                    yield self.env.timeout(time1)

                self.buffers[1].store.put(self)  # Buffer 2
                
                if self.production_system.twin_system == None:
                        loggerTwin.debug(f"{self.name}, entered buffer2 at  {self.env.now}")
            
                else: 
                    logger.debug(f"{self.name}, entered buffer2 at  {self.env.now}")

                # print("Buffer 2", len(buffers[1].store.items))
                with self.machines[1].isResource.request() as request:  # Machine 1
                    yield request
                    self.buffers[1].store.get()
                    # print("{} on Machine 2 at {}".format(
                    #     self.name, self.env.now))
                    # yield self.env.timeout(generate_random_machines(MU_Process, self.si_process, self.production_system))
                    if self.production_system.twin_system == None:
                        loggerTwin.debug(f"{self.name}, started being processed on Machine2 at  {self.env.now}")
            
                    else: 
                        logger.debug(f"{self.name}, started being processed on Machine2 at  {self.env.now}")
    
                    if self.production_system.use_seeds:
                        time2 = max(0, self.production_system.r2.uniform(
                            self.processing_time_range[0], self.processing_time_range[1]))
                    else:
                        time2 = max(0, np.random.uniform(
                            self.processing_time_range[0], self.processing_time_range[1]))

                    # Update processing time average for m2
                    self.production_system.update_processing_time(time2, 1)

                    # time2 = self.production_system.r2.lognormal(
                    #     MU_Process, self.si_process)

                    # with open(self.production_system.name + " M2" + ".csv", "a") as f:
                    #     f.write(str(time1) + "\n")

                    yield self.env.timeout(time2)

                # print("{} released into Finished product Inventory at {}".format(self.name, self.env.now))
                self.shop_e = self.env.now    # Order Exited shop floor
                if self.production_system.twin_system == None:
                    loggerTwin.debug(f"{self.name}, Exited the production system at  {self.env.now}")
        
                else: 
                    logger.debug(f"{self.name}, Exited the production system at  {self.env.now}")
                
                self.production_system.parts_produced += 1
                # Update throughput time
                # self.production_system.update_throughput_time(
                #     self.shop_e - self.production_system.previous_exit_time)
                self.production_system.previous_exit_time = self.shop_e
                # updates state average Flow_time
                # self.production_system.update_flow_time_state_elements(
                #     self.shop_e - self.shop_s)
                # updates flow time file
                if self.production_system.files:
                    self.production_system.track_flow_time(self.env, self)

                self.done = True

            else:
                # print(self.name, " is verifying that wip  is ", Wip())
                self.env.step()


def generate_random_machines(mean, std, production_system):
    value = abs(random.gauss(mean, std))
    # with open(production_system.name + ".csv", "a") as f:
    #     f.write(str(value) + "\n")
    return value


def generate_random_parts(mean, std, production_system):
    value = abs(random.gauss(mean, std))
    # with open(production_system.name + ".csv", "a") as f:
    #     f.write(str(value) + "\n")
    return value
