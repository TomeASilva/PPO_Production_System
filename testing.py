# from agents import PPO, Agent, FixedPolicy
# import unittest
# import numpy as np

# production_system_configuration = {
#     "processing_time_range": (0, 20),
#     "initial_wip_cap": np.random.randint(1, 100),
#     "decision_epoch_interval": 30,
#     "track_state_interval": 5,
#     }

# ppo_networks_configuration = {"trunk_config": {"layer_sizes": [100, 100],
#                                       "activations": ["relu", "relu"]},

#                      "mu_head_config": {"layer_sizes": [1],
#                                         "activations": ["relu"]},
#                      "cov_head_config": {"layer_sizes": [1],
#                                         "activations": ["sigmoid"]},
#                      "critic_net_config": {"layer_sizes": [100, 64, 1],
#                                             "activations": ["relu", "relu", "linear"]},
#                      "input_layer_size": 7
#                         }
# ppo_configuration = {
#     "gamma": 0.99
# }

# def _aggregate_episodes(old_agregate, new_episode):

#     states_agregate, actions_agregate, qsa_agregate = old_agregate
    
#     states, actions, next_states, rewards, qsa = new_episode
    
#     states_agregate = np.vstack((states_agregate, states))
#     actions_agregate = np.vstack((actions_agregate, actions_agregate))
#     qsa_agregate = np.vstack((qsa_agregate, qsa))
#     old_agregate = (states_agregate, actions_agregate, qsa_agregate)

#     return old_agregate
    
# myAgent = Agent("MyFairLady",
#                 action_range=(1, 100),
#                 conwip=1000,
#                 **ppo_configuration,
#                 production_system_configuration=production_system_configuration,
#                 ppo_networks_configuration = ppo_networks_configuration,
#                 current_number_episodes=None,
#                 total_number_episodes=None,
#                 episode_queue=None,
#                 warm_up_length=30,
#                 run_length=500)

# if __name__ == "__main__":
    
#     myAgent.FixedPolicy = FixedPolicy(myAgent.conwip, myAgent.action_range)
#     myAgent.PPO = PPO(**ppo_networks_configuration, action_range=myAgent.action_range)
#     myAgent.PPO.build_models()
#     myAgent.collect_episodes(1)
#     states, actions, next_states, rewards, qsa = myAgent.buffer2.unroll_memory(myAgent.gamma)
    
#     aggregate_episodes = (states, actions, qsa)
    
#     print(f"First Episode {aggregate_episodes}")
#     myAgent.collect_episodes(1)
    
#     states, actions, next_states, rewards, qsa = myAgent.buffer2.unroll_memory(myAgent.gamma)
#     episode = (states, actions, next_states, rewards, qsa)
#     print(f"New episode {episode}")
#     aggregate_episodes = _aggregate_episodes(aggregate_episodes, episode)
#     print(f"aggregate_episodes: {aggregate_episodes}")
    
    
    
    # print(f"states: {states}")
    # print(f"shape states: {states.shape}")
    # print(f"actions: {actions}")
    # print(f" actions: {actions.shape}")
    # print(f"next_states: {next_states}")
    # print(f" next_states: {next_states.shape}")
    # print(f"rewards: {rewards}")
    # print(f" rewards: {rewards.shape}")
    # print(f"qsa: {qsa}")
    # print(f" qsa: {qsa.shape}")
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(0.001, decay=0.01)

