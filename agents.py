import simpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from buffer import EpBuffer
from production_system import ProductionSystem
from collections import deque
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from production_system import logger, loggerTwin, logger_state
from multiprocessing import Manager, Queue, Process
from multiprocessing.queues import Full
import multiprocessing
import csv
import datetime
import logging
import pickle
gradient_logger = logging.getLogger(__name__)
gradient_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler("./logs/gradient.log")
file_handler.setFormatter(formatter)
gradient_logger.addHandler(file_handler)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
# print(tf.__version__)

def summarize_performance(path):
    wip = np.loadtxt(path + "/" + " WIP.csv", delimiter=";", unpack=False)
    lead_time = np.loadtxt(path + "/" + " flow_time.csv",
                           delimiter=";", unpack=False, usecols=(1, 2, 3))

    throughput = np.loadtxt(
        path + "/" + " Parts Produced.csv", delimiter=";", unpack=False)
    wip = np.mean(wip[:, 1])
    flow_time = np.mean(lead_time[:, -2])
    lead_time = np.mean(lead_time[:, -1])
    parts_produced = throughput[-1, 1]
    throughput = throughput[-1, 1] / 3000

    print("Average Lead Time: ", lead_time)
    print("Average Flow Time: ", flow_time)
    print("Average WIP: ", wip)
    print("Average Throughput", throughput)
    print("Parts Produced", parts_produced)
    # print("Total reward", production_system.sum_rewards)

    return (lead_time, flow_time, wip, throughput, parts_produced)

def build_networks(layer_sizes, activations, input):
    num_layers = len(layer_sizes)
    output = keras.layers.Dense(units=layer_sizes[0], activation=activations[0], kernel_initializer='glorot_normal')(input)
    for i in range(1, num_layers):
        output = keras.layers.Dense(units=layer_sizes[i], activation=activations[i], kernel_initializer='glorot_normal')(output)

    return output


def build_model(input, output, name):
    return keras.Model(input, output, name=name)


class BasePolicy():
    def __init__(self, action_range):
        self.action_range = action_range


class PPO(BasePolicy):
    def __init__(self,
                 agent_name,
                 input_layer_size,
                 trunk_config,
                 mu_head_config,
                 cov_head_config,
                 critic_net_config,
                 action_range, 
                 summary_writer=None):
        """
        Inputs:
        input_layer_size -> int
        trunk_config -> dict {"layer_sizes": [123, 123, 123 .....]}
        mu_head_config -> dict {"layer_sizes": [123, 123, 123 .....]}
        critic_net_config -> dict {"layer_sizes": [123, 123, 123 .....]}
        action_range -> tuple (0, 100)
        """

        BasePolicy.__init__(self, action_range)

        self.input_layer_size = input_layer_size
        self.trunk_config = trunk_config
        self.mu_head_config = mu_head_config
        self.cov_head_config = cov_head_config
        self.critic_net_config = critic_net_config
        self.name = f"WIP set by PPO"
        self.action_range = action_range
        self.agent_name = agent_name
        self.summary_writer = summary_writer
        self.number_action_calls = 0
    def build_models(self):

        self.input = keras.Input(shape=(self.input_layer_size), name="state", dtype=tf.float32)
        self.trunk = build_networks(**self.trunk_config, input=self.input)
        mu_head = build_networks(**self.mu_head_config, input=self.trunk)
        cov_head = build_networks(**self.cov_head_config, input=self.trunk)
        critic = build_networks(**self.critic_net_config, input=self.input)
        
        # Creates a model for mu cov and critic
        self.actor_mu = build_model(self.input, mu_head, "actor_mu")
        self.actor_cov = build_model(self.input, cov_head, "actor_cov")
        self.critic = build_model(self.input, critic, "critic")
        #---Start Find the variables of the actor cov model
        actor_cov_n_layers_head = len(self.cov_head_config["layer_sizes"])
        actor_cov_total_n_layers = len(self.actor_cov.layers)
        cov_head_variables = []
        
        for i in range (actor_cov_total_n_layers - actor_cov_n_layers_head, actor_cov_total_n_layers, 1):
            for variable in self.actor_cov.get_layer(index = i).trainable_variables:
                cov_head_variables.append(variable)
        #---END Find the variables of the actor cov model
        self.cov_head_variables = cov_head_variables
        # ---START Creates a way to access the  current value of the weights of the networks
        # actor_mu and actor_cov will be identical with exception of the output layer
        self.current_parameters = {"mu": [variable.numpy() for variable in self.actor_mu.trainable_variables],
                           "cov": [variable.numpy() for variable in cov_head_variables],
                           "critic": [variable.numpy() for variable in self.critic.trainable_variables]
                           }
        # ---END Creates a way to access the  current value of the weights of the networks
        # ---START Creates a way to acess the variables of the models (used to apply the gradients)
        self.variables = {"mu": self.actor_mu.trainable_variables,
                        "cov": cov_head_variables,
                        "critic": self.critic.trainable_variables}
        # ---END Creates a way to acess the variables of the models (used to apply the gradients)

        if self.agent_name == "Global Agent":
          
            self.trunk_old = build_networks(**self.trunk_config, input=self.input)
            mu_head_old = build_networks(**self.mu_head_config, input=self.trunk_old)
            cov_head_old = build_networks(**self.cov_head_config, input=self.trunk_old)
            self.actor_mu_old = build_model(self.input, mu_head_old, "actor_mu_old")
            self.actor_cov_old = build_model(self.input, cov_head_old, "actor_cov_old")
            
            #---Start Find the variables of the actor cov model
            actor_cov_n_layers_head_old = len(self.cov_head_config["layer_sizes"])
            actor_cov_total_n_layers_old = len(self.actor_cov_old.layers)
            cov_head_variables_old = []
            for i in range (actor_cov_total_n_layers_old - actor_cov_n_layers_head_old, actor_cov_total_n_layers_old, 1):
                for variable in self.actor_cov_old.get_layer(index = i).trainable_variables:
                    cov_head_variables_old.append(variable)
        #---END Find the variables of the actor cov model
            self.cov_head_variables_old = cov_head_variables_old
            self.variables_old =  {"mu": self.actor_mu_old.trainable_variables,
                            "cov": cov_head_variables_old}
            
            self.current_parameters_old =  {"mu": [variable.numpy() for variable in self.actor_mu_old.trainable_variables],
                           "cov": [variable.numpy() for variable in cov_head_variables_old],
                           "critic": [variable.numpy() for variable in self.critic.trainable_variables]
                           }


    def get_action(self, state):
        """
        Inputs:
        state -> numpy.ndarray

        Returns:
        action -> int
        mu -> tf.Tensor([[0.4335983]], shape=(1, 1), dtype=float32)

        cov -> tf.Tensor([[0.39956307]], shape=(1, 1), dtype=float32)

       """
        assert state.shape == (1, self.input_layer_size), "Check the dimensions of State"
        # ---START Generates the average and standard deviation for the wip at the given stage
        mu = self.actor_mu(state)
        cov = self.actor_cov(state)
        
        # print(f"Average: {mu}")
        # print(f"Std: {cov}")
        # ---END Generates the average and standard deviation for the wip at the given stage

        # Computes a noram distribution
        probability_density_func = tfp.distributions.Normal(mu, cov)
        # Samples a WIP from the distribution
        action = probability_density_func.sample()

        action = tf.clip_by_value(action, self.action_range[0], self.action_range[1])
        
        if self.summary_writer != None:
             with self.summary_writer.as_default():
                 tf.summary.histogram("Mu dist_" + self.agent_name, mu, self.number_action_calls)
                 tf.summary.histogram("Cov dist_" + self.agent_name, cov, self.number_action_calls)
                 tf.summary.histogram("Action_" + self.agent_name, action, self.number_action_calls)
                 
        # print(f"Action: {action}")
        self.number_action_calls += 1
        return int(action)

    def get_state_value(self, state):
        state_value = self.critic(state)
        return float(state_value)

class RandomPolicy(BasePolicy):

    def __init__(self, action_range):
        BasePolicy.__init__(self, action_range)
        self.name = f"WIP set to random uniform dist"

    def get_action(self):
        return np.random.randint(self.action_range[0], self.action_range[1])


class FixedPolicy(BasePolicy):

    def __init__(self, wip, action_range):

        BasePolicy.__init__(self, action_range)
        self.wip = wip
        self.name = f"WIP set to {self.wip}"

    def get_action(self, state):
        """
        state -> np.ndarray  dummy not being used
        """
        return self.wip


class Agent:
    def __init__(self,
                 name,
                 action_range,
                 conwip,
                 production_system_configuration,
                 ppo_networks_configuration,
                 gamma,
                 current_number_episodes,
                 total_number_episodes,
                 episode_queue,
                 warm_up_length,
                 run_length,
                 epsilon,
                 record_statistics,
                 gradient_steps_per_episode,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 parameters_queue,
                 actor_optimizer_mu,
                 actor_optimizer_cov,
                 critic_optimizer,
                 entropy,
                 number_episodes_worker
                 
                 ):
        
        self.name = name
        self.action_range = action_range
        self.conwip = conwip
        self.buffer1 = EpBuffer()
        self.buffer2 = EpBuffer()
        self.production_system_configuration = production_system_configuration
        self.current_number_episodes = current_number_episodes  # number of episodes already run
        self.total_number_episodes = total_number_episodes  # Total number of episodes to be run
        self.episode_queue = episode_queue
        self.warm_up_length = warm_up_length
        self.run_length= run_length
        self.ppo_networks_configuration = ppo_networks_configuration
        self.gamma = gamma
        self.action_range = action_range
        self.episode_queue = episode_queue
        self.epsilon = epsilon
        self.record_statistics = record_statistics
        self.gradient_steps_per_episode = gradient_steps_per_episode
        self.gradient_clipping_actor = gradient_clipping_actor
        self.gradient_clipping_critic = gradient_clipping_critic
        self.parameters_queue = parameters_queue
        self.actor_optimizer_mu = actor_optimizer_mu
        self.actor_optimizer_cov = actor_optimizer_cov
        self.critic_optimizer = critic_optimizer
        self.entropy = entropy
        self.number_episodes_worker = number_episodes_worker
        
    def collect_episodes(self,
                        number_ep,
                        files,
                        logging):

        for ep in range(number_ep):
            random_seed_1 = np.random.randint(1, 100000)
            random_seed_2 = np.random.randint (1, 100000)
            random_seed_3 = np.random.randint (1, 100000)
            random_seeds = (random_seed_1, random_seed_2, random_seed_3)
            
            env = simpy.Environment()  # can it be forked

            production_system_1 = ProductionSystem(env=env,
                             **self.production_system_configuration,
                             ep_buffer=self.buffer1,
                             policy=self.FixedPolicy, # will be defined only after instatiation of child classes
                             warmup_time=self.warm_up_length,
                             use_seeds=True,
                             files=files,
                             random_seeds= random_seeds,
                             logging=logging,
                             run_length=self.run_length
                             
                             )
            production_system_2 = ProductionSystem(env=env,
                                                   **production_system_configuration,
                                                   ep_buffer=self.buffer2,
                                                   twin_system=production_system_1,
                                                   policy=self.PPO, # will be defined only after instatiation of child classes
                                                   warmup_time=self.warm_up_length,
                                                   use_seeds=True,
                                                   files=files,
                                                   random_seeds=random_seeds,
                                                   logging=logging,
                                                   run_length=self.run_length
                                                  )

        env.run(until=self.warm_up_length + self.run_length)

        logger.debug("Simulation Finished")
        loggerTwin.debug("Simulation Finished")
        logger_state.debug(f"Simulation Finished")            
        
        if files:
            return (production_system_1.path, production_system_2.path)


class GlobalAgent(Agent):
    def __init__(self,
                 name,
                 action_range,
                 conwip,
                 production_system_configuration,
                 ppo_networks_configuration,
                 gamma,
                 current_number_episodes,
                 total_number_episodes,
                 episode_queue,
                 warm_up_length,
                 run_length,
                 epsilon,
                 record_statistics,
                 number_of_child_agents,
                 number_episodes_worker,
                 gradient_steps_per_episode,
                 save_checkpoints,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 parameters_queue,
                 average_reward_queue,
                 actor_optimizer_mu,
                 actor_optimizer_cov,
                 critic_optimizer,
                 entropy
                 ):

                
        Agent.__init__(self,
                 name=name,
                 action_range = action_range,
                 conwip = conwip,
                 production_system_configuration = production_system_configuration,
                 current_number_episodes = current_number_episodes,
                 total_number_episodes=total_number_episodes,
                 episode_queue = episode_queue,
                 ppo_networks_configuration=ppo_networks_configuration,
                 epsilon=epsilon,
                 record_statistics=record_statistics,
                 warm_up_length=warm_up_length,
                 run_length=run_length,
                 gamma=gamma,
                 gradient_steps_per_episode=gradient_steps_per_episode,
                 gradient_clipping_actor=gradient_clipping_actor,
                 gradient_clipping_critic=gradient_clipping_critic,
                 parameters_queue=parameters_queue,
                 actor_optimizer_mu=actor_optimizer_mu,
                 actor_optimizer_cov=actor_optimizer_cov,
                 critic_optimizer=critic_optimizer,
                 entropy=entropy,
                 number_episodes_worker = number_episodes_worker)

        self.number_of_child_agents = number_of_child_agents
        self.save_checkpoints = save_checkpoints
        self.average_reward_queue = average_reward_queue
        self.rewards = deque(maxlen=100)

    def restore_old_models(self, ppo_config):
        try:
            self.PPO.actor_mu_old.load_weights("./saved_checkpoints/actor_mu/")
            self.PPO.actor_cov_old.load_weights("./saved_checkpoints/actor_cov/")
            self.PPO.critic.load_weights("./saved_checkpoints/critic/")
            
            self.PPO.actor_mu.load_weights("./saved_checkpoints/actor_mu/")
            self.PPO.actor_cov.load_weights("./saved_checkpoints/actor_cov/")
            
            actor_cov_n_layers_head = len(ppo_config["mu_head_config"]["layer_sizes"])
            actor_cov_total_n_layers = len(self.PPO.actor_cov.layers)
            cov_head_variables = []

            for i in range (actor_cov_total_n_layers - actor_cov_n_layers_head, actor_cov_total_n_layers, 1):
                for variable in self.PPO.actor_cov.get_layer(index = i).trainable_variables:
                    cov_head_variables.append(variable)

            self.PPO.cov_head_variables = cov_head_variables
            
            self.PPO.current_parameters = {"mu": [variable.numpy() for variable in self.PPO.actor_mu.trainable_variables],
                           "cov": [variable.numpy() for variable in cov_head_variables],
                           "critic": [variable.numpy() for variable in self.PPO.critic.trainable_variables]
                           }

            #check if variables in fact were loaded
            # for key, model in self.PPO.current_parameters.items():
            #     for variable in model:
            #         tf.print(variable)        
            
            cov_head_variables_old = []

            for i in range (actor_cov_total_n_layers - actor_cov_n_layers_head, actor_cov_total_n_layers, 1):
                for variable in self.PPO.actor_cov_old.get_layer(index = i).trainable_variables:
                    cov_head_variables_old.append(variable)

            self.PPO.cov_head_variables_old = cov_head_variables_old
            
            self.PPO.current_parameters_old = {"mu": [variable.numpy() for variable in self.PPO.actor_mu_old.trainable_variables],
                           "cov": [variable.numpy() for variable in cov_head_variables_old],
                           "critic": [variable.numpy() for variable in self.PPO.critic.trainable_variables]
                           }

            return "Loading models was succefull"
        except Exception as message :
            return (message)
    def training_loop(self):
        try:
            #---START Create a summary writer
            if self.record_statistics: 
                self.writer = tf.summary.create_file_writer(f"./summaries/global/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
            self.current_pass = 0
            #---END of Create summary writer
            self.FixedPolicy = FixedPolicy(self.conwip, self.action_range)
            self.PPO = PPO(**self.ppo_networks_configuration, action_range=self.action_range, agent_name=self.name, summary_writer= self.writer)
            self.PPO.build_models() #build models (Critic, Actor Networks) 
            print(f"1 optimization cycle corresponds to {self.number_of_child_agents * self.number_episodes_worker} episodes and {self.gradient_steps_per_episode} gradient steps")
            
        
             #---START Load variable weights if self.save_checkpoints is activated
            if self.save_checkpoints:
                try:
                    with open("./saved_checkpoints/optimizers/actor/actor_optimizer_mu.pkl", "rb") as file:
                        self.actor_optimizer_mu = pickle.load(file)
                    
                    
                    with open("./saved_checkpoints/optimizers/actor/actor_optimizer_cov.pkl", "rb") as file:
                        self.actor_optimizer_cov = pickle.load(file)
                    
                    with open("./saved_checkpoints/optimizers/critic/critic_optimizer.pkl", "rb") as file:
                        self.critic_optimizer = pickle.load(file)
                
                except Exception as e :    
                    print(e)
                    print("We couldn't load the optimizer state")
                    print("The optimizers state will be reseted")
                    
                
                message = self.restore_old_models(self.ppo_networks_configuration)
                print(message, "\n" )
            
           
                
            #---END Load variable weights if self.save_checkpoints is activated
            self.number_optimization_cycles = 0
            self.number_of_gradient_descent_steps = 0
            # for every optimizatin cycle there will be un x episodes and y number of gradient descent steps
            while self.current_number_episodes.value < self.total_number_episodes:
                # because workers run in pararell the number of episodes 
                #would change within a cycle 
                
                for i in range(self.number_of_child_agents):
                    #Put enough parameters for all workers
                    self.parameters_queue.put(self.PPO.current_parameters_old, block=True, timeout=30)
            
                    
                #---START Record the values for the weights of policy gradient NN
                if self.record_statistics:
                    with self.writer.as_default():
                        for key, parameters in self.PPO.variables.items():
                            for variable in parameters:
                                tf.summary.histogram(f"Params_{self.name}_{str(key)}_{variable.name}", variable, self.number_optimization_cycles)
                #---END Record the values for the weights of policy gradient NN 
                
    
                # Collect all the episodes waiting on the queue
                for i in range(self.number_of_child_agents * self.number_episodes_worker):
                    episode = self.episode_queue.get(block=True, timeout=30)

                    if i == 0:
                        states, actions, next_states, rewards, qsa = episode
                    
                    else:
                        states_temp, actions_temp, next_states_temp, rewards_temp, qsa_temp = episode
                        
                        states = np.vstack((states, states_temp))
                        actions = np.vstack((actions, actions_temp))
                        qsa = np.vstack((qsa, qsa_temp))
                #---END collect episodes available from all worker
                
                
                   #---START gradient descent for actor Nets
                for i in range(self.gradient_steps_per_episode):
                    
                    
                    #---START Record the values for the weights of policy gradient NN
                    if self.record_statistics:
                        with self.writer.as_default():
                            for key, parameters in self.PPO.variables.items():
                                for variable in parameters:
                                    tf.summary.histogram(f"Params_{self.name}_{str(key)}_{variable.name}", variable, self.number_of_gradient_descent_steps)
                    #---END Record the values for the weights of policy gradient NN 
                  
                    gradients, entropy = self.gradient_actor(states, actions, qsa)
                    

                    #---START Cliping gradients
                    for key, gradient in gradients.items():
                        gradients[key] = [tf.clip_by_value(value, -self.gradient_clipping_actor, self.gradient_clipping_actor) for value in gradient]
                    #---END Clipping Gradients
                    
                    #---START Record gradient to summaries
                    if self.record_statistics:
                        with self.writer.as_default():
                                for key, gradient_list in gradients.items():
                                    for gradient, variable in zip(gradient_list, self.PPO.variables[key]):
                                
                                        tf.summary.histogram(f"Gradients_{self.name}_{str(key)}_{variable.name}", gradient, self.number_of_gradient_descent_steps)
                    #---END Record gradient to summaries
                     
                    #---Start Record average entropy for episodes
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar(f"Entropy", entropy, self.number_of_gradient_descent_steps)
                    #---End Record average entropy'for episodes
                    
                    #---START apply gradients for actor
                    for key, value in gradients.items():
                        if key == "mu":
                            self.actor_optimizer_mu.apply_gradients(zip(value, self.PPO.variables[key]))
                        if key == "cov":
                            self.actor_optimizer_cov.apply_gradients(zip(value, self.PPO.variables[key]))
                            
                    #---END apply gradients for actor
                  
                    #---END gradient descent for actor Nets
                    
                    
                    
                    
                    #---START gradient descent for critic
                    critic_gradient = self.gradient_critic(states, qsa)
                        #---START Gradient Clipping critic
                    critic_gradient = [tf.clip_by_value(value, -self.gradient_clipping_critic, self.gradient_clipping_critic) for value in critic_gradient]
                        #---
                    #---START Record gradient to summaries
                    if self.record_statistics:
                        with self.writer.as_default():
                            for gradient, variable in zip(critic_gradient, self.PPO.variables["critic"]):
                                tf.summary.histogram(f"Gradients_{self.name}_critic_{variable.name}", gradient, self.number_of_gradient_descent_steps)

                    #---END Record gradient to summaries
                    #---START Aplly Critic Gradients
                    self.critic_optimizer.apply_gradients(zip(critic_gradient, self.PPO.variables["critic"]))
                    #---
                    #---START Observe the value of given state to see convergence
                    state = np.array([10, 3, 1, 0.80, 0.50, 18, 15])
                    value = self.state_value(state.reshape(1, -1))
                    value = float(value.numpy()[0])
                
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar(f"State_value", value, self.number_of_gradient_descent_steps)
                    #---END Observe the value of given state to see convergence
                    self.number_of_gradient_descent_steps += 1
                #---END gradient descent for critic
                
                
                
                
                #---START update self.current_parameter with the parameters resulting from n steps of gradient descent
                self.current_parameters = {"mu": [variable.numpy() for variable in self.PPO.actor_mu.trainable_variables],
                            "cov": [variable.numpy() for variable in self.PPO.cov_head_variables],
                            "critic": [variable.numpy() for variable in self.PPO.critic.trainable_variables]
                            }
                #---END update self.current_parameter with the parameters resulting from n steps of gradient descent
                #---START Update Old policy seting theta_old = theta
                for key, value in self.PPO.current_parameters.items():
                    if key != "critic":
                        for n, variable in enumerate(self.PPO.variables_old[key]):
                            variable.assign(value[n])
                #---END Update Old policy seting theta_old = theta
                
                #---START updattrigger = True self.current_parameter_old with 
                self.current_parameters_old = self.current_parameters
                
                
                self.number_optimization_cycles += 1
                
               
                #---START after n iterations RUN EPISODE and PRINT REWARD
                if self.number_optimization_cycles % 1 == 0:
                    rewards_volatile = []
                    for i in range (1):
                        path_twin, path_PPO = self.collect_episodes(1, True, False)
                        
                        print("--------------PPO Policy---------------")
                        summarize_performance(path_PPO)
                        
                        print("--------------No WIP Cap--------------")
                        summarize_performance(path_twin)
      
                        
                        _, _, _, rewards_temp, _ = self.buffer2.unroll_memory(self.gamma)
                        reward = np.sum(rewards_temp)
                        rewards_volatile.append(reward)
                    # av_reward = sum(rewards_volatile) / len(rewards_volatile)
                    # max_reward = max(rewards_volatile)
                    # min_reward = min(rewards_volatile)
                    print(f"Ep number: {self.current_number_episodes.value}: Reward : {reward}")
                    
                    # print(f"Ep number: {self.self.current}: Average: {av_reward} -- Max: {max_reward} -- Min {min_reward} ")
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar(f"Reward", reward, self.current_number_episodes.value)
                            # tf.summary.scalar(f"Max_Reward", max_reward, self.self.current_number_episodes.value)
                            # tf.summary.scalar(f"Min_Reward", min_reward, self.self.current_number_episodes.value)
                    # self.rewards.append(av_reward)
                #---END after n iterations of the loop run episode and print reward
                
                #---START Save weights at current iter
            
                if self.save_checkpoints:
                    
                    with open("./saved_checkpoints/optimizers/actor/actor_optimizer_mu.pkl", "wb") as file:
                        pickle.dump(self.actor_optimizer_mu, file)
                    
                    with open("./saved_checkpoints/optimizers/actor/actor_optimizer_cov.pkl", "wb") as file:
                        pickle.dump(self.actor_optimizer_cov, file)
                    
                    
                    with open("./saved_checkpoints/optimizers/critic/critic_optimizer.pkl", "wb") as file:
                        pickle.dump(self.critic_optimizer, file)
    
                        
                    self.PPO.actor_mu.save_weights("./saved_checkpoints/actor_mu/")
                    self.PPO.actor_cov.save_weights("./saved_checkpoints/actor_cov/")
                    self.PPO.critic.save_weights("./saved_checkpoints/critic/")
                    
            
            #--- END Main RL loop
            #---After all cycles run a summary of the training session and write it on a file
            with open("Running_Log.csv", "a") as file:
                    
                writer = csv.writer(file, delimiter=",")
                writer.writerow(["Run", self.current_number_episodes.value])
            rewards_volatile = []
            for i in range (1):
                path_twin, path_PPO = self.collect_episodes(1, True, False)
                
                print("--------------PPO Policy---------------")
                summarize_performance(path_PPO)
                
                print("--------------No WIP Cap--------------")
                summarize_performance(path_twin)
                        
                
                _, _, _, rewards_temp, _ = self.buffer2.unroll_memory(self.gamma)
                reward = np.sum(rewards_temp)
                rewards_volatile.append(reward)
            # av_reward = sum(rewards_volatile) / len(rewards_volatile)
            # max_reward = max(rewards_volatile)
            # min_reward = min(rewards_volatile)
            print(f"Ep number: {self.current_number_episodes.value}: Reward : {reward}")
            
            # for key, model in self.current_parameters.items():
            #     for variable in model:
            #         tf.print(variable)
                    
            print(f"Exited Global Agent")
            
        except KeyboardInterrupt:
            rewards_volatile = []
            for i in range (1):
                path_twin, path_PPO = self.collect_episodes(1, True, False)
                
                print("--------------PPO Policy---------------")
                summarize_performance(path_PPO)
                
                print("--------------No WIP Cap--------------")
                summarize_performance(path_twin)                    
                
                _, _, _, rewards_temp, _ = self.buffer2.unroll_memory(self.gamma)
                reward = np.sum(rewards_temp)
                rewards_volatile.append(reward)
            # av_reward = sum(rewards_volatile) / len(rewards_volatile)
            # max_reward = max(rewards_volatile)
            # min_reward = min(rewards_volatile)
            print(f"Ep number: {self.current_number_episodes.value}: Reward : {reward}")

            print("Press ctr + C one last time. Summary has be saved!")
            # self.average_reward_queue.put(sum(self.rewards) / len(self.rewards), block=True, timeout=30)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 7]),))
    def state_value(self, state):
        value = self.PPO.critic(state)
        return value
    
    # @tf.function(input_signature=(tf.TensorSpec(shape=[None, 7]), tf.TensorSpec(shape=[None, 1]), tf.TensorSpec(shape=[None, 1])))  
    def gradient_actor(self, states, actions, Qsa):
        with tf.GradientTape(persistent=True) as tape:
            gradient_logger.debug(f"Cycle {self.number_optimization_cycles}")
            #---START Actor gradient calculation
            #---START Get the parameters for the Normal dist
            gradient_logger.debug(f"Actions")
            for value in actions:
                gradient_logger.debug(value)

            mu = self.PPO.actor_mu(states)
            gradient_logger.debug(f"MU")
            for value in mu:
                gradient_logger.debug (value)
            
            cov = self.PPO.actor_cov(states)
            gradient_logger.debug(f"Cov")
            for value in cov: 
                gradient_logger.debug(value)
            
            mu_old = tf.stop_gradient(self.PPO.actor_mu_old(states))
            cov_old = tf.stop_gradient(self.PPO.actor_cov_old(states))
            
            gradient_logger.debug(f"MU_old")
            for value in mu_old:
                gradient_logger.debug (value)
            
            gradient_logger.debug(f"Cov_old")
            for value in cov_old:
                gradient_logger.debug (value)
            
            #---END Get the parameters for the Normal dist
            #---START Advantage function computation and normalization
            advantage_function = Qsa - self.PPO.critic(states)
            advantage_function_mean = tf.math.reduce_mean(advantage_function)
            advantage_function_std = tf.math.reduce_std(advantage_function)
            advantage_function = tf.math.divide(advantage_function - advantage_function_mean, (advantage_function_std + 1.0e-8))
            #---END Advantage function computation and normalization
            #---START compute the Normal distributions
            self.probability_density_func = tfp.distributions.Normal(mu, cov)
            self.probability_density_func_old = tfp.distributions.Normal(mu_old, cov_old)
            #---END compute the Normal distributions
            #---Entropy
            entropy = self.probability_density_func.entropy()
            entropy_average = tf.reduce_mean(entropy)
            
            #---
            #---START compute the probability of the actions taken at the current episode
            probs = self.probability_density_func.prob(actions)
            
            gradient_logger.debug(f"Probs")
            for value in probs:
                gradient_logger.debug(value)
            
            probs_old = tf.stop_gradient(self.probability_density_func_old.prob(actions))
            
            
            gradient_logger.debug(f"Probs_old")
            for value in probs_old:
                gradient_logger.debug(value)
            
            #---START Ensemble Actor loss function
            self.probability_ratio = tf.math.divide(probs + 1e-5, probs_old + 1e-5)
            
            gradient_logger.debug(f"Probability ratio")
            for value in self.probability_ratio:
                gradient_logger.debug(value)
                
            cpi = tf.math.multiply(self.probability_ratio, tf.stop_gradient(advantage_function))
            clip = tf.math.minimum(cpi, tf.multiply(tf.clip_by_value(self.probability_ratio, 1 - self.epsilon, 1 + self.epsilon), tf.stop_gradient(advantage_function)))
            actor_loss = -tf.reduce_mean(clip) - entropy_average * self.entropy
            gradient_logger.debug(f"Action Loss: {actor_loss}")
            
            #---END Ensemble Actor loss function
        
        #---START Compute gradients for average
        gradients_mu = tape.gradient(actor_loss, self.PPO.actor_mu.trainable_variables)
        #---
        gradient_logger.debug(f"Gradients Mu")
        for value in gradients_mu:
            gradient_logger.debug(value)
        
        #---START Compute gradients for the covariance

        gradients_cov = tape.gradient(actor_loss, self.PPO.cov_head_variables)
        # END Compute gradients for the covariance
        
        gradient_logger.debug("Gradient Cov")
        for value in gradients_cov:
            gradient_logger.debug(value)
            
        gradients = {"mu": gradients_mu,
                     "cov": gradients_cov,}
        #---END Actor gradient calculation
        
          
        return gradients, entropy_average
            
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 7]), tf.TensorSpec(shape=[None, 1])))
    def gradient_critic(self, states, Qsa):
        with tf.GradientTape(persistent=True) as tape:
        
            critic_cost = tf.losses.mean_squared_error(Qsa, self.PPO.critic(states))
    
        gradients_critic = tape.gradient(critic_cost, self.PPO.critic.trainable_variables)
    
        return gradients_critic
            
class WorkerAgent(Agent):
    def __init__(self,
                 name,
                 action_range, 
                 conwip, 
                 production_system_configuration,
                 ppo_networks_configuration,
                 gamma,
                 current_number_episodes, 
                 total_number_episodes,
                 episode_queue,
                 number_episodes_worker,
                 warm_up_length,
                 run_length,
                 epsilon,
                 record_statistics,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 parameters_queue,
                 actor_optimizer_mu,
                 actor_optimizer_cov,
                 critic_optimizer,
                 entropy,
                 gradient_steps_per_episode
                 ):
        Agent.__init__(self,
                        name=name,
                        action_range=action_range, 
                        conwip=conwip, 
                        production_system_configuration=production_system_configuration,
                        current_number_episodes=current_number_episodes, 
                        total_number_episodes=total_number_episodes,
                        episode_queue=episode_queue,
                        warm_up_length=warm_up_length,
                        run_length=run_length,
                        epsilon=epsilon,
                        gamma=gamma,
                        record_statistics=record_statistics,
                        ppo_networks_configuration=ppo_networks_configuration,
                        gradient_clipping_actor=gradient_clipping_actor,
                        gradient_clipping_critic=gradient_clipping_critic,
                        parameters_queue=parameters_queue,
                        actor_optimizer_mu=actor_optimizer_mu,
                        actor_optimizer_cov=actor_optimizer_cov,
                        critic_optimizer=critic_optimizer,
                        entropy=entropy,
                        number_episodes_worker=number_episodes_worker,
                        gradient_steps_per_episode=gradient_steps_per_episode
                        )

        
     
        
        
    
    def training_loop(self):
         # ---START build NETworks for critic and actor
        if self.record_statistics:
            self.writer = tf.summary.create_file_writer(f"./summaries/global/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.name}")
        
        self.FixedPolicy = FixedPolicy(self.conwip, self.action_range)
        self.PPO = PPO(**ppo_networks_configuration, action_range=self.action_range, agent_name=self.name,summary_writer=self.writer)
        self.PPO.build_models() #build models (Critic, Actor Networks)
        # ---

        
        while self.current_number_episodes.value < self.total_number_episodes:
            self.number_episodes_run_upuntil = self.current_number_episodes.value
            # ---Update variables with information coming from gradient descent
            self.update_variables()
            # ---
            
            # ---START Collect n episodes from this worker
            for ep in range(self.number_episodes_worker): # Run more than episode per iteration of PPO
                if self.current_number_episodes.value < self.total_number_episodes:
                    self.collect_episodes(self.number_episodes_worker, False, False)
                    
                    states, actions, next_states, rewards, qsa = self.buffer2.unroll_memory(self.gamma)
                    rollout = (states, actions, next_states, rewards, qsa)
                    try:
                        self.episode_queue.put(rollout, block=False)
                    except Full:
                        print("Queue Was full")
                        break
                        
                    self.current_number_episodes.value += 1
                else: break
                
            #---END Collect n episodes from this worker   
              
        print(f"Exited {self.name}")
            
    def update_variables(self):
         #---Get current parameters from queue(put in by the global agent)
        try:
            self.new_params = self.parameters_queue.get(block=True)
        except Exception as e:
            print(e)
        
        #---
        #---START assign the variables of the worker with the variable values from global
        for key, value in self.new_params.items():
            for n, variable in enumerate(self.PPO.variables[key]):
                variable.assign(value[n])
        #END assign the variables of the worker with the variable values from global

        
        #---START update variable current_parameters to reflect the information provided by global
        self.current_parameters = {"mu": [variable.numpy() for variable in self.PPO.actor_mu.trainable_variables],
                        "cov": [variable.numpy() for variable in self.PPO.cov_head_variables],
                        "critic": [variable.numpy() for variable in self.PPO.critic.trainable_variables]
    
                        }
        #---END update variable current_parameters to reflect the information provided by global
    

production_system_configuration = {
    "processing_time_range": (0, 20),
    "initial_wip_cap": np.random.randint(1, 100),
    "decision_epoch_interval": 30,
    "track_state_interval": 5,
    }

ppo_networks_configuration = {"trunk_config": {"layer_sizes": [100, 100],
                                      "activations": [tf.nn.leaky_relu, tf.nn.leaky_relu]},

                     "mu_head_config": {"layer_sizes": [64, 32, 1],
                                        "activations": [tf.nn.leaky_relu, tf.nn.leaky_relu, "softplus"]},
                     "cov_head_config": {"layer_sizes": [64, 32, 1],
                                        "activations": [tf.nn.leaky_relu, tf.nn.leaky_relu, "softplus"]},
                     "critic_net_config": {"layer_sizes": [100, 64, 1],
                                            "activations": ["relu", "relu", "linear"]},
                     "input_layer_size": 7
                        }

hyperparameters = {"ppo_networks_configuration" : ppo_networks_configuration,
                   "actor_optimizer_mu": tf.keras.optimizers.Adam(learning_rate=0.0001),
                   "actor_optimizer_cov": tf.keras.optimizers.Adam(learning_rate=0.0001),
                    "critic_optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001),
                    "entropy":0.09,
                    "gamma":0.999,
                    "gradient_clipping_actor": 0.8, 
                    "gradient_clipping_critic": 0.8, 
                    "gradient_steps_per_episode": 5,
                    "epsilon": 0.2,
                    "number_episodes_worker": 1
                    }
    
agent_config = {
    "action_range": (0, 100),
    "total_number_episodes" : 3,
    "conwip": 10000,
    "warm_up_length": 100,
    "run_length": 3000
    
}
if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn')
    number_of_workers = 1

    params_queue = Manager().Queue(number_of_workers)
    current_number_episodes = Manager().Value("i", 0)    
    episode_queue = Manager().Queue(number_of_workers*hyperparameters["number_episodes_worker"])
    average_reward_queue = Queue(1)
    global_agent = GlobalAgent(**hyperparameters,
                               **agent_config,
                               production_system_configuration = production_system_configuration,
                               name="Global Agent",
                               record_statistics=True,
                               average_reward_queue=average_reward_queue,
                               number_of_child_agents=number_of_workers,
                               save_checkpoints=True,
                               episode_queue=episode_queue,
                               current_number_episodes=current_number_episodes,
                               parameters_queue=params_queue)
    workers = []
    for _ in range(number_of_workers):
        print("worker created")
        myWorker = WorkerAgent(**hyperparameters,
                        **agent_config,
                        production_system_configuration=production_system_configuration,
                        name=f"Worker_{_}",
                        record_statistics=True,
                        episode_queue=episode_queue,
                        current_number_episodes=current_number_episodes,
                        parameters_queue=params_queue) 
        
        workers.append(myWorker)
   


    processes = []
    p1 = Process(target=global_agent.training_loop)
    processes.append(p1)
    p1.start()
    for worker in workers:
        
        p = Process(target=worker.training_loop)
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    print(average_reward_queue.get())
    print("Simulation Run")
        
        

        
        
        
        
