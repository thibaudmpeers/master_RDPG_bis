import numpy as np
import sys
import pygame
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from Environment.LocalAgent import LocalAgent

resolution = (750, 750)
middle_screen = int(resolution[0] / 2), int(resolution[0] / 2)
white = pygame.Color(255, 255, 255)
black = pygame.Color(0, 0, 0)
radius = 300
width = 30
band = 1
agent_radius = 10


class LoopFollowingAgents:
    def __init__(self, available_obs, mode_action, obs_u_i=False, type_action=None, number_agents=None,
                 random_start=None, delay=0.1, dt=0.025, d_ref=2, local_time_step=0.0125,
                 active_agents=None, gains_reward=None, u_i=True, stacked_frames=False, obs_front=False,
                 reward_analysis=False, perturbations=False, load_agent=None, number_steps=1000):

        self.available_obs = available_obs
        self.mode_action = mode_action
        self.obs_u_i = False if obs_u_i is None else obs_u_i
        self.type_action = [True, True, False] if type_action is None else type_action
        self.N = 6 if number_agents is None else number_agents
        self.number_steps = number_steps
        # 0 => good position, 1 => good pos + rand(-1,1), otherwise => random
        self.random_start = 1 if random_start is None else random_start
        self.action_agents = [0] if active_agents is None else active_agents
        self.nb_active_agents = len(self.action_agents)
        self.gains = [1, 1, 0.25, 10] if gains_reward is None else gains_reward
        self.reward_analysis = reward_analysis
        self.length_env = number_agents * d_ref
        self.d_ref = d_ref
        self.local_dt = local_time_step
        self.max_speed = 2
        self.steps_action = int(round(dt / self.local_dt))
        if self.steps_action == 0:
            print('Invalid parameter dt and local_time_step')

        self.dt = self.steps_action * self.local_dt
        self.steps_delay = int(round(delay / self.dt)) + 1

        # computation of B
        tmp1 = np.eye(self.N)
        tmp2 = np.concatenate((np.eye(self.N - 1), np.zeros((self.N - 1, 1))), axis=1)
        tmp3 = np.concatenate((np.zeros((1, self.N)), tmp2))
        tmp4 = tmp1 - tmp3
        self.B = tmp4
        self.B[0][-1] = -1

        self.local_agents = []
        for i in range(self.N):
            if i == load_agent:
                self.local_agents.append(LocalAgent(steps_delay=self.steps_delay, local_time_step=self.local_dt,
                                                    max_speed=self.max_speed, length_env=self.length_env,
                                                    steps_action=self.steps_action, u_i=u_i,
                                                    perturbations=perturbations, load_disturbance=True))
            else:
                self.local_agents.append(LocalAgent(steps_delay=self.steps_delay, local_time_step=self.local_dt,
                                                    max_speed=self.max_speed, length_env=self.length_env,
                                                    steps_action=self.steps_action, u_i=u_i,
                                                    perturbations=perturbations))

        if self.mode_action == 0:
            self.action_space = 1
        elif self.mode_action == 1:
            self.action_space = self.type_action.count(True)
            if self.action_space == 0:
                print('error: no action on term P, I or D selected')
                sys.exit()
        else:
            self.action_space = self.type_action.count(True)
            if self.action_space == 0:
                print('error: no action on term P, I or D selected')
                sys.exit()

        self.action_space *= self.nb_active_agents
        self.one_action = True if (self.action_space == 1) else False
        nb_obs_agent = 2 if not obs_u_i else 3  # observation number of one agent
        if available_obs == 1:
            # agent knows the observations of all the agents
            self.obs_state = nb_obs_agent * self.N
        elif available_obs == 2:
            # agent knows his observations and the observations of the behind agent
            self.obs_state = nb_obs_agent * 2
        else:
            # agent knows only his observations
            self.obs_state = nb_obs_agent
            self.available_obs = 0

        self.stacked_frames = stacked_frames
        if self.stacked_frames:
            if self.available_obs == 0:
                self.obs_front = obs_front
                if self.obs_front:
                    self.observation_space = self.obs_state + 2 * (self.steps_delay - 1)
                else:
                    self.observation_space = self.obs_state + self.steps_delay - 1

            elif self.available_obs == 2:
                self.observation_space = self.obs_state + 3 * (self.steps_delay - 1)
            else:
                self.observation_space = self.obs_state + self.N * (self.steps_delay - 1)
        else:
            self.observation_space = self.obs_state

        self.behind_agent = [obs_agent + 1 if obs_agent != self.N - 1 else 0 for obs_agent in self.action_agents]
        self.front_agent = [obs_agent - 1 if obs_agent != 0 else self.N - 1 for obs_agent in self.action_agents]

        self.delayed_error_positions = None
        self.delayed_modified_error_positions = None
        self.state = None
        self.saved_u = None
        self.save_trajectory = None
        self.info_trajectory = None
        self.first_render = True
        self.screen = None
        self.color = []
        # print(self.d_ref)
        # print(self.B)
        # print(init_position)
        # print(init_error_position)
        # self.counter = 0

    def error_computation(self, positions, speed=True, mod_ref=True, action=0):

        error_positions = -np.matmul(self.B, positions)
        error_positions -= self.d_ref
        # print(error_positions[0])
        for i in range(self.N):
            if abs(error_positions[i]) > self.length_env/2:
                error_positions[i] -= self.length_env*np.sign(error_positions[i])
        error_positions[self.action_agents] += action
        if not speed:
            return error_positions

        if mod_ref:
            delayed_error = self.delayed_modified_error_positions
            self.delayed_modified_error_positions = error_positions
        else:
            delayed_error = self.delayed_error_positions
            self.delayed_error_positions = error_positions

        error_speeds = (error_positions - delayed_error) / (self.local_dt * self.steps_action)
        return error_positions, error_speeds

    def initialization(self, initial_state):
        init_positions = np.zeros(self.N)
        if initial_state is None:
            for i in range(self.N):
                if self.random_start != 2:
                    init_positions[i] = -i * self.d_ref + (self.N - 1 / 2) * self.d_ref
                    if self.random_start == 1:
                        init_positions[i] += (np.random.rand() - 0.5) * self.d_ref
                else:
                    init_positions[i] = np.random.rand() * self.N * self.d_ref
            init_positions = -np.sort(init_positions)

            init_speeds = np.zeros(self.N)
            for i in range(len(init_speeds)):
                init_speeds[i] = (np.random.rand() - 0.5) * self.max_speed
        else:
            init_positions = initial_state['positions'].copy()
            init_speeds = initial_state['speeds'].copy()

        init_error_positions = self.error_computation(positions=init_positions, speed=False)
        error_speeds = init_speeds - np.concatenate((init_speeds[-1], init_speeds[0:self.N - 1]), axis=None)

        self.delayed_error_positions = init_error_positions  # - self.local_dt*self.steps_action*error_speeds
        self.delayed_modified_error_positions = init_error_positions - self.local_dt * self.steps_action * error_speeds

        return init_positions, init_error_positions, init_speeds, error_speeds

    def reset(self, initial_state=None, get_init_state=False, save_traj=False):
        self.save_trajectory = save_traj

        self.info_trajectory = {'positions': np.zeros((self.number_steps, self.N)),
                                'speeds': np.zeros((self.number_steps, self.N)),
                                'control_input': np.zeros((self.number_steps, self.nb_active_agents)),
                                'action': np.zeros((self.number_steps, self.action_space)),
                                'error_position_learning': np.zeros((self.number_steps, self.nb_active_agents)),
                                'error_speed_learning': np.zeros((self.number_steps, self.nb_active_agents)),
                                'int_term': np.zeros((self.number_steps, self.nb_active_agents)),
                                'reward': np.zeros((self.number_steps, self.nb_active_agents, 4)) if
                                self.reward_analysis else np.zeros(self.number_steps)} if \
            save_traj else None
        init_positions, init_error_positions, init_speeds, init_error_speeds = self.initialization(
            initial_state=initial_state)

        init_u_i = np.zeros(self.N)
        for i in range(self.N):
            self.local_agents[i].reset(init_position=init_positions[i], init_error=init_error_positions[i],
                                       init_speed=init_speeds[i])
        self.state = {'positions': init_positions, 'error_positions': init_error_positions, 'speeds': init_speeds,
                      'error_speeds': init_error_speeds, 'int_terms': init_u_i}
        if self.stacked_frames:
            self.saved_u = np.zeros((self.steps_delay - 1, self.N))

        init_state = {'positions': init_positions, 'speeds': init_speeds}
        if get_init_state:
            return [self._get_obs(), init_state]
        return self._get_obs()

    def step(self, action, current_step):  # action is the variation of d_ref on the agent 0

        actions_u = np.zeros((self.nb_active_agents, 2))
        action_ref = np.zeros(self.nb_active_agents)
        actions_gains = np.zeros((self.nb_active_agents, 3))
        if self.mode_action == 0:
            action_ref = action
        elif self.mode_action == 1:
            index_action = 0
            for j in range(self.nb_active_agents):
                for i in range(len(self.type_action)):
                    if self.type_action[i]:
                        actions_u[j][i] = action if self.one_action else action[index_action]
                        index_action += 1
        else:
            index_action = 0
            for j in range(self.nb_active_agents):
                for i in range(len(self.type_action)):
                    if self.type_action[i]:
                        actions_gains[j][i] = action if self.one_action else action[index_action]
                        index_action += 1

        if self.save_trajectory:
            if self.one_action:
                self.info_trajectory['action'][current_step][0] = action
            else:
                self.info_trajectory['action'][current_step] = action.copy()

        new_control_inputs = np.zeros(self.N)
        new_u_i = np.zeros(self.N)
        new_speeds = np.zeros(self.N)
        new_positions = np.zeros(self.N)
        positions = self.state['positions']
        speeds = self.state['speeds']

        error_positions, error_speeds = self.error_computation(positions=positions, action=action_ref)

        index_active_agent = 0
        for j in range(self.N):
            if j in self.action_agents:
                new_positions[j], new_speeds[j], new_control_inputs[j], new_u_i[j] = self.local_agents[j].take_step(
                    error_position=error_positions[j], error_speed=error_speeds[j],
                    actions_u=actions_u[index_active_agent], actions_gains=actions_gains[index_active_agent],
                    enable_PID=True)
                index_active_agent += 1
                # print('position ', j,' at time ',i,' = ', self.positions[j], ' at counter = ', self.counter)
            else:
                new_positions[j], new_speeds[j], new_control_inputs[j], new_u_i[j] = self.local_agents[j].take_step(
                    error_position=error_positions[j], error_speed=error_speeds[j])

        if self.stacked_frames:
            if self.steps_delay != 1:
                to_save_u = np.expand_dims(new_control_inputs, axis=0)
                self.saved_u = np.concatenate((to_save_u, self.saved_u[:-1]), axis=0)
            else:
                self.saved_u = new_control_inputs

        new_error_positions, new_error_speeds = self.error_computation(positions=new_positions, mod_ref=False)

        reward = self.reward(u_agents=new_control_inputs[self.action_agents],
                             error_positions=new_error_positions[self.action_agents],
                             error_speeds=new_error_speeds[self.action_agents])

        self.state['positions'] = new_positions
        self.state['error_positions'] = new_error_positions
        self.state['speeds'] = new_speeds
        self.state['error_speeds'] = new_error_speeds
        self.state['int_terms'] = new_u_i

        done = False
        # if(self.episode_step >= self.max_steps):
        #     done = True
        if self.save_trajectory:
            self.info_trajectory['positions'][current_step] = positions
            self.info_trajectory['speeds'][current_step] = speeds
            self.info_trajectory['control_input'][current_step] = new_control_inputs[self.action_agents]
            self.info_trajectory['error_position_learning'][current_step] = error_positions[self.action_agents]
            self.info_trajectory['error_speed_learning'][current_step] = error_speeds[self.action_agents]
            self.info_trajectory['int_term'][current_step] = new_u_i[self.action_agents]
            self.info_trajectory['reward'][current_step] = reward

        return self._get_obs(), reward, done, self.info_trajectory

    def reward(self, u_agents, error_positions, error_speeds):
        alpha = self.gains[0]
        beta = self.gains[1]
        gam = self.gains[2]
        reward = [] if self.reward_analysis else 0
        for u, err_p, err_s in zip(u_agents, error_positions, error_speeds):
            first_term = -alpha * err_p ** 2
            second_term = -beta * err_s ** 2
            third_term = -gam * u ** 2
            # first_term = -alpha*abs(error_position)
            # second_term = -beta*abs(error_speed)
            # third_term = -gamma*abs(u)
            fourth_term = 0
            if np.abs(err_p + self.d_ref) < 0.2:
                fourth_term = -self.gains[3]
            if self.reward_analysis:
                if self.nb_active_agents == 1:
                    reward = [first_term, second_term, third_term, fourth_term]
                else:
                    reward.append([first_term, second_term, third_term, fourth_term])
            else:
                reward += first_term + second_term + third_term + fourth_term
        return reward

    def _get_obs(self):
        error_positions = self.state['error_positions']
        error_speeds = self.state['error_speeds']
        u_i = self.state['int_terms']

        all_observations = []
        for a_agent, f_agent, b_agent in zip(self.action_agents, self.front_agent, self.behind_agent):
            if self.available_obs == 0:
                error_learning_agent = np.asarray([error_positions[a_agent], error_speeds[a_agent]])
                if not self.obs_u_i:
                    observation_agent = error_learning_agent
                else:
                    observation_agent = np.concatenate((error_learning_agent, u_i[a_agent]), axis=None)

            elif self.available_obs == 1:
                if not self.obs_u_i:
                    observation_agent = np.concatenate((error_positions, error_speeds), axis=None)
                else:
                    observation_agent = np.concatenate((error_positions, error_speeds, u_i), axis=None)

            else:
                if not self.obs_u_i:
                    learning_agent_obs = \
                        np.concatenate((error_positions[a_agent], error_speeds[a_agent]), axis=None)
                    behind_obs = np.concatenate((error_positions[b_agent], error_speeds[b_agent]), axis=None)
                else:
                    learning_agent_obs = np.concatenate(
                        (error_positions[a_agent], error_speeds[a_agent], u_i[a_agent]), axis=None)
                    behind_obs = np.concatenate(
                        (error_positions[b_agent], error_speeds[b_agent], u_i[b_agent]), axis=None)
                observation_agent = np.concatenate((learning_agent_obs, behind_obs), axis=None)

            if self.stacked_frames:
                if self.available_obs == 0:
                    if self.obs_front:
                        obs_action = self.saved_u[:, [a_agent, f_agent]]
                    else:
                        obs_action = np.expand_dims(self.saved_u[:, a_agent], axis=0)
                elif self.available_obs == 1:
                    obs_action = self.saved_u
                else:
                    obs_action = self.saved_u[:, [a_agent, f_agent, b_agent]]

                observation_agent = np.concatenate([observation_agent, obs_action], axis=None)
            all_observations.append(observation_agent)

        if self.nb_active_agents == 1:
            return all_observations[0]
        return all_observations

    def feedback_params(self):
        return self.local_agents[0].feedback_params()

    def render(self):
        if self.first_render:
            self.first_render = False
            pygame.init()
            self.screen = pygame.display.set_mode(resolution)
            tmp = np.zeros((self.N, 10))
            p = plt.plot(tmp)
            for i in range(self.N):
                self.color.append(to_rgba(p[i].get_color()))
            del p
            plt.clf()
        self.screen.fill(white)
        pygame.draw.circle(self.screen, black, middle_screen, radius + width + band)
        pygame.draw.circle(self.screen, white, middle_screen, radius + width - band)
        pygame.draw.circle(self.screen, black, middle_screen, radius - width + band)
        pygame.draw.circle(self.screen, white, middle_screen, radius - width - band)

        positions = self.state['positions']
        angle = positions*2*np.pi/self.length_env

        screen_positions = np.zeros((2, self.N))
        screen_positions[0] = np.cos(angle)*radius + middle_screen[0]
        screen_positions[1] = -np.sin(angle)*radius + middle_screen[1]

        screen_positions = screen_positions.astype(int)

        for i in range(self.N):
            color_agent = pygame.Color(int(self.color[i][0]*255), int(self.color[i][1]*255), int(self.color[i][2]*255))
            pygame.draw.circle(self.screen, color_agent, screen_positions[:, i], agent_radius)

        pygame.display.flip()
