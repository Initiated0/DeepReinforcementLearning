import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear


from configs.q4_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        
        print("image height: ", img_height)
        print("image width: ", img_width)
        print("image #of channels: ", n_channels)
        print("image history length: ", self.config.state_history)
        print("#of actions: ", num_actions)
        
        pad_1L = ( (4-1) * img_height - 4 + 8) // 2
        pad_2L = ( (2-1) * img_height - 2 + 4) // 2
        pad_3L = ( (1-1) * img_height - 2 + 4) // 2
# image height:  8
# image width:  8
# image #of channels:  6
# image history length:  4
# #of actions:  5
# Initializing parameters randomly
# Evaluating...



        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################
        # Define the Q network
        self.q_network = nn.Sequential(
            nn.Conv2d(n_channels*self.config.state_history, 32, kernel_size=8, stride=4, padding=pad_1L),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=pad_2L),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=pad_3L),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Define the Target network with the same architecture
        self.target_network = nn.Sequential(
            nn.Conv2d(n_channels*self.config.state_history, 32, kernel_size=8, stride=4, padding=pad_1L),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=pad_2L),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=pad_3L),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None
        # state = torch.flatten(state, start_dim=1)
        # print(state.shape)
        # torch.Size([1, 8, 8, 24])


        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ############
        if network == 'q_network':
            out = self.q_network(state.permute(0, 3, 2, 1).float())
            # out = self.q_network(state.float())
        elif network == 'target_network':
            out = self.target_network(state.permute(0, 3, 1, 2).float())
            # out = self.target_network(state.float())
        else:
            raise ValueError('Invalid network name: {}'.format(network))
        return out
        ##############################################################
        ######################## END YOUR CODE #######################
        


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
