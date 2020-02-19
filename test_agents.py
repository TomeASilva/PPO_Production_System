from agents import PPO
import agents
import unittest
import numpy as np

# ppo_configuration = {"trunk_config": {"layer_sizes": [100, 100],
#                                       "activations": ["relu", "relu"]},
                     
#                      "mu_head_config": {"layer_sizes":[1], 
#                                         "activations": ["relu"]},
#                      "cov_head_config":{ "layer_sizes":[1],
#                                         "activations": ["sigmoid"]}, 
#                      "critic_net_config" : {"layer_sizes":[100, 64, 1],
#                                             "activations": ["relu", "relu", "linear"]}
#                         }

# PPO_policy = PPO(input_layer_size = 8, 
#                  action_range=(0, 100),
#                  **ppo_configuration)

# PPO_policy.build_models()
# state = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])

# # value = PPO_policy.actor_mu(state)
# a, b , c = PPO_policy.get_action(state)

class TestPolicies(unittest.TestCase):
    
    def setUp(self):
        self.action_range = (0, 1)
        self.ppo_configuration = {"trunk_config": {"layer_sizes": [100, 100],
                                      "activations": ["relu", "relu"]},
                     
                     "mu_head_config": {"layer_sizes":[1], 
                                        "activations": ["relu"]},
                     "cov_head_config":{ "layer_sizes":[1],
                                        "activations": ["sigmoid"]}, 
                     "critic_net_config" : {"layer_sizes":[100, 64, 1],
                                            "activations": ["relu", "relu", "linear"]}
                        }
        
        self.input_layer_size = 8
        
    def test_BasePolicy(self):
        basepolicy = agents.BasePolicy(self.action_range)
        self.assertEqual(basepolicy.action_range, (0, 1))
    
    def test_RandomPolicy(self):
        random_policy = agents.RandomPolicy(self.action_range)
        wip = random_policy.get_action()
        self.assertGreaterEqual(wip, self.action_range[0])
        self.assertLessEqual(wip, self.action_range[1]) 
    def test_PPO(self):
        ppo = agents.PPO(input_layer_size= self.input_layer_size, **self.ppo_configuration, action_range=self.action_range)       

        

    
        

if __name__ == '__main__':
    unittest.main()



