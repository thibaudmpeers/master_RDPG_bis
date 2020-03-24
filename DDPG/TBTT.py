import torch
from torch.nn.functional import smooth_l1_loss


class TBTT_critic:
    def __init__(self, critic_model, T, optimizer, steps_traj, gamma):
        self.critic_model = critic_model
        self.T = T
        self.optimizer = optimizer
        self.steps_traj = steps_traj
        self.gamma = gamma

    def train(self, batch_histoy):
        observations_hist, actions_hist, rewards_hist, inputs_hist, target_values_hist, count = batch_histoy
        init_state = self.critic_model.init_hidden_state(count)
        states = [(None, init_state)]

        inputs_hist.requires_grad = True
        actions_hist.requires_grad = True

        critic_losses = torch.zeros(1)

        mean_values = torch.zeros(1)

        # max_grad = torch.zeros(0)

        for j in range(self.steps_traj):
            # obs = observations_hist[:, j]
            # action = actions_hist[:, j]
            # reward = rewards_hist[:, j]
            inputs = inputs_hist[:, j]
            target_value = target_values_hist[:, j]
            reward = rewards_hist[:, j]
            action = actions_hist[:, j]

            target = reward + self.gamma * target_value

            state = states[-1][1].detach()
            state.requires_grad = True
            value, next_state = self.critic_model(inputs, action, state)
            states.append((state, next_state))

            while len(states) > self.T:
                del states[0]

            critic_loss = smooth_l1_loss(target, value)

            mean_value = value.mean().detach()
            mean_values += mean_value / self.steps_traj

            critic_losses += critic_loss.detach()

            self.optimizer.zero_grad()

            critic_loss.backward(retain_graph=True)

            # print('hey = ', state.grad.abs().max())

            for i in range(self.T - 1):
                if states[-i - 2][0] is None:
                    break
                current_grad = states[-i - 1][0].grad
                # print(i, current_grad.max())
                states[-i - 2][1].backward(current_grad, retain_graph=True)

            self.optimizer.step()

        return critic_losses.cpu().numpy(), mean_values


class TBTT_actor:
    def __init__(self, actor_model, T, optimizer, steps_traj, critic_model):
        self.actor_model = actor_model
        self.T = T
        self.optimizer = optimizer
        self.steps_traj = steps_traj
        self.critic_model = critic_model

    def train(self, batch_history):
        observations_hist, actions_hist, rewards_hist, inputs_hist, target_values_hist, count = batch_history

        init_actor_state = self.actor_model.init_hidden_state(count)
        init_critic_state = self.critic_model.init_hidden_state(count)

        states = [(None, init_actor_state)]
        critic_state = init_critic_state

        inputs_hist.requires_grad = True

        actor_losses = torch.zeros(1)

        for j in range(self.steps_traj):

            inputs = inputs_hist[:, j]

            state = states[-1][1].detach()
            state.requires_grad = True
            action, next_state = self.actor_model(inputs, state)
            states.append((state, next_state))
            value, critic_state = self.critic_model(inputs, action, critic_state.detach())

            actor_loss = -value.mean()
            actor_losses += actor_loss.detach()

            self.optimizer.zero_grad()

            actor_loss.backward(retain_graph=True)

            for i in range(self.T - 1):
                if states[-i - 2][0] is None:
                    break
                current_grad = states[-i - 1][0].grad
                states[-i - 2][1].backward(current_grad, retain_graph=True)

            self.optimizer.step()

        return actor_losses.cpu().numpy()
