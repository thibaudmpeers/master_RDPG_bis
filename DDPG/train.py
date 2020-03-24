from __future__ import division
import torch
import torch.nn.functional as F
import DDPG.utils as utils
import DDPG.model as model
from DDPG.TBTT import TBTT_critic, TBTT_actor


def get_nn_arch_params():
    return model.return_nn_arch_params()


class Trainer:

    def __init__(self, state_dim, action_dim, steps_traj, ram, device, learning_rate=1e-5, tau=1e-5, decay_tau=0):

        self.batch_size = 128
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.init_tau = tau
        self.tau = self.init_tau
        self.decay_factor_tau = decay_tau
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ram = ram
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = model.Actor(self.state_dim, self.action_dim).to(device)
        self.target_actor = model.Actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)

        self.critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

        T = 8

        self.TBTT_critic = TBTT_critic(self.critic, T, self.critic_optimizer, steps_traj, self.gamma)
        self.TBTT_actor = TBTT_actor(self.actor, T, self.actor_optimizer, steps_traj, self.critic)

    def get_exploitation_action(self, inputs, h_actor):
        action, h_actor = self.actor.forward(inputs, h_actor)
        return action.detach(), h_actor

    def get_exploration_action(self, inputs, h_actor):
        std = 2
        action, h_actor = self.actor.forward(inputs, h_actor)
        # new_action = action.data.numpy() + (self.noise.sample() * self.action_lim.numpy())
        new_action = action.detach() + (torch.tensor(self.noise.sample()).float().to(self.device) * std)
        return new_action, h_actor

    def optimize(self):
        batch_history = self.ram.sample()

        critic_loss, mean = self.TBTT_critic.train(batch_history)
        actor_loss = self.TBTT_actor.train(batch_history)

        utils.soft_update(self.target_actor, self.actor, self.tau)
        utils.soft_update(self.target_critic, self.critic, self.tau)

        return critic_loss, actor_loss, mean

    def save_models(self, episode_count, path):
        torch.save(self.target_actor.cpu().state_dict(), path + 'Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.cpu().state_dict(), path + 'Models/' + str(episode_count) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, episode, path_folder):
        self.actor.load_state_dict(torch.load(path_folder + 'Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(path_folder + 'Models/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('Models loaded successfully')

    def return_params(self):
        return [self.batch_size, self.learning_rate, self.gamma, self.init_tau, self.decay_factor_tau]

    def get_models_params(self):
        nb_actor_params = 0
        for parameter in self.actor.parameters():
            nb_params_layer = 1
            for nb_nn in list(parameter.size()):
                nb_params_layer = nb_params_layer * nb_nn
            nb_actor_params += nb_params_layer

        nb_critic_params = 0
        for parameter in self.critic.parameters():
            nb_params_layer = 1
            for nb_nn in list(parameter.size()):
                nb_params_layer = nb_params_layer * nb_nn
            nb_critic_params += nb_params_layer
        return nb_actor_params, nb_critic_params

    def update_tau(self, episode_number):
        self.tau = self.init_tau/(1 + self.decay_factor_tau*episode_number)

    # def get_learning_rate(self):
    #     for param in self.actor_optimizer.param_group:
    #         print(param['lr'])
