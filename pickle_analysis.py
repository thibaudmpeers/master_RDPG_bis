import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from Environment.LoopFollowingAgents import LoopFollowingAgents
from DDPG.train import Trainer
import torch

all_tests = True
mode_loop = True

if all_tests:
    number_agents = ['6', '7', '8']
    # number_agents = ['4', '10']
    # observation = ['0', '0+front_agent', '1', '2']
    observation = ['1']
    # learning_rate = ['-1', '-2', '-3', '-4', '-5', '-6']
    learning_rate = ['-6']
    tau = ['-1', '-2', '-3', '-4', '-5']
    # tau = ['-2']
    gamma = ['0.99', '0.998']
    # test_names = ['test2']
    test_names = ['test1', 'test2', 'test3', 'test4']
    decay_tau_factor = ['0', '10e-1']
else:
    number_agents = ['6']
    observation = ['0', '0+front_agent', '2']
    # observation = ['0']
    learning_rate = ['-5']
    tau = ['-5']
    gamma = ['0.99']
    test_names = ['test1']
    decay_tau_factor = ['0', '10e-1']
    # decay_tau_factor = ['150e-1']


n_ex = 20


def mean(array):
    return sum(array) / len(array)


def variance(array):
    m = mean(array)
    return np.sqrt(mean([(x - m) ** 2 for x in array]))


def save_reward(folder_path, from_pickle):
    number_episodes = from_pickle['number_episodes']
    # number_episodes = 100
    number_steps = from_pickle['number_steps']
    # print(number_steps)
    n = from_pickle['number_agents']
    random_start = from_pickle['random_start']
    delay = from_pickle['delay']
    dt = from_pickle['time_step']
    d_ref = from_pickle['distance_ref']
    local_dt = from_pickle['local_time_step']
    gains_reward = from_pickle['gains_reward']
    u_i = from_pickle['enable_u_i']

    all_reward_rl = from_pickle['sum_reward_history']
    reward_rl = []
    nor_reward_rl = []
    reward_no_delay = []
    nor_reward_no_delay = []
    reward_no_action = []
    reward_no_control = []

    initial_states = from_pickle['initial_states']

    env_no_control = LoopFollowingAgents(available_obs=0, mode_action=2, type_action=[True, True, True],
                                         number_agents=n, random_start=random_start, delay=delay, dt=dt, d_ref=d_ref,
                                         local_time_step=local_dt, gains_reward=gains_reward, u_i=u_i,
                                         number_steps=number_steps)
    env_no_action = LoopFollowingAgents(available_obs=0, mode_action=2, type_action=[True, True, True], number_agents=n,
                                        random_start=random_start, delay=delay, dt=dt, d_ref=d_ref,
                                        local_time_step=local_dt, gains_reward=gains_reward, u_i=u_i,
                                        number_steps=number_steps)
    env_no_delay = LoopFollowingAgents(available_obs=0, mode_action=2, type_action=[True, True, True], number_agents=n,
                                       random_start=random_start, delay=0, dt=dt, d_ref=d_ref,
                                       local_time_step=local_dt, gains_reward=gains_reward, u_i=u_i,
                                       number_steps=number_steps)

    [kp, ki, kd] = env_no_control.feedback_params()

    action = [-kp, -ki, -kd]
    # print('action = ', action)

    for episode in range(0, number_episodes, n_ex):
        print(episode)
        initial_state = initial_states[episode]
        _ = env_no_control.reset(initial_state)
        _ = env_no_action.reset(initial_state)
        _ = env_no_delay.reset(initial_state)

        sum_reward1 = 0
        sum_reward2 = 0
        sum_reward3 = 0

        for step in range(number_steps):
            _, reward1, _, infos = env_no_control.step(action, step)
            _, reward2, _, _ = env_no_action.step([0, 0, 0], step)
            _, reward3, _, _ = env_no_delay.step([0, 0, 0], step)
            sum_reward1 += reward1
            sum_reward2 += reward2
            sum_reward3 += reward3
            # print('1 = ',sum_reward1)
            # print('2 = ',sum_reward2)
            # print('3 = ',sum_reward3)

        reward_no_control.append(sum_reward1)
        reward_no_action.append(sum_reward2)
        reward_no_delay.append(sum_reward3)
        reward_rl.append(all_reward_rl[episode])

        # print(sum_reward1)
        # print(sum_reward2)
        # print(sum_reward3)

        nor_reward_no_delay.append((sum_reward3 - sum_reward2) / (sum_reward1 - sum_reward2))
        nor_reward_rl.append((all_reward_rl[episode] - sum_reward2) / (sum_reward1 - sum_reward2))

    episodes = range(0, number_episodes, n_ex)
    # print('mean no control = ', mean(reward_no_control))
    # print('mean no action = ', mean(reward_no_action))
    # print('mean no delay = ', mean(reward_no_delay))
    # print('mean RL = ', mean(reward_rl))

    f1 = plt.figure(1)
    # plt.plot(episodes, reward_rl)
    plt.plot(episodes, reward_rl, episodes, reward_no_delay, episodes, reward_no_control, episodes,
             reward_no_action)
    plt.legend(('RL', 'no delay', 'no control', 'no action'))
    # f1.show()
    # f2 = plt.figure(2)
    # plt.plot(episodes, nor_reward_rl, episodes, nor_reward_no_delay)
    # plt.legend(('RL', 'no delay'))
    # f2.show()
    # plt.show()
    save_path = folder_path + 'reward_history_all_RL.png'
    f1.set_size_inches(20, 14)
    f1.savefig(save_path)
    # plt.show()
    f1.clf()
    # f2.clf()


def plot_learning_graphs(folder_path, from_pickle):
    reward_history = from_pickle['sum_reward_history']
    value_loss = from_pickle['critic_loss']
    policy_loss = from_pickle['actor_loss']

    # mean_ratio = information['mean_ratio']
    # std_ratio = information['std_ratio']

    f1 = plt.figure(1)
    plt.plot(reward_history)
    # f1.show()

    f3 = plt.figure(3)
    plt.plot(value_loss)

    f4 = plt.figure(4)
    plt.plot(policy_loss)

    save_path = folder_path + '/reward_history.png'
    f1.set_size_inches(20, 14)
    f1.savefig(save_path)
    f1.clf()
    save_path = folder_path + '/value_loss_history.png'
    f3.set_size_inches(20, 14)
    f3.savefig(save_path)
    f3.clf()
    save_path = folder_path + '/policy_loss_history.png'
    f4.set_size_inches(20, 14)
    f4.savefig(save_path)
    f4.clf()


def save_trajectory(folder_path, from_pickle):
    infos = from_pickle['infos']
    # print('length infos = ', len(infos))
    nb_episode = len(infos) - 1
    infos = infos[nb_episode]
    episode_number = from_pickle['saved_episode_number'][nb_episode]
    #
    episode_steps = from_pickle['number_steps']
    dt = from_pickle['time_step']
    d_ref = from_pickle['distance_ref']
    n = from_pickle['number_agents']
    active_agents = from_pickle['active_agents']
    # sum_reward_history = from_pickle['sum_reward_history']
    mode_action = from_pickle['mode_action']
    type_action = from_pickle['type_action']
    # print('mode_actor_critic = ', from_pickle['mode_actor_critic'])
    # execution_times = from_pickle['execution_times']
    # print(execution_times)
    del from_pickle

    # f1 = plt.figure(1)
    # plt.plot(execution_times)
    # f1.show()
    save_traj = True
    if save_traj:

        reward = infos['reward']
        positions = infos['positions']
        speed = infos['speeds']
        error_speeds_learning = infos['error_speed_learning']
        error_learning = infos['error_position_learning']
        action = infos['action']
        control_learning = infos['control_input']
        u_i = infos['int_term']
        # print('HEY')
        # print(len(reward))
        # print(len(reward[0]))
        # print('sum reward = ', np.sum(reward))

        time = np.linspace(dt, episode_steps * dt, episode_steps)
        for epi in range(episode_steps - 1):
            for j in range(n):
                if np.abs(positions[epi + 1][j] - positions[epi][j]) > d_ref * (n - 1):
                    positions[epi + 1][j] = np.nan

        f2, ax = plt.subplots(4, 2)
        title = 'episode number ' + str(episode_number) + ', mode action ' + str(mode_action) + ', type action = ' \
                + str(type_action) + ', active agents ' + str(active_agents)
        f2.suptitle(title)
        ax[0][0].plot(time, positions)
        ax[0][0].set_title('Agent Positions')
        ax[0][1].plot(time, speed)
        ax[0][1].set_title('Agent Speeds')
        ax[1][0].plot(time, reward)
        ax[1][0].set_title('Reward')
        ax[1][1].plot(time, error_learning)
        ax[1][1].set_title('Error Position of the Learning Agent')
        ax[2][0].plot(time, error_speeds_learning)
        ax[2][0].set_title('Error Speed of the Learning Agent')
        ax[2][1].plot(time, action)
        ax[2][1].set_title('action')
        ax[3][0].plot(time, control_learning)
        ax[3][0].set_title('Control Input of the Learning Agent')
        ax[3][1].plot(time, u_i)
        ax[3][1].set_title('integral term of the learning agent')

        ax[0][0].axis([0, episode_steps * dt, 0, d_ref * n])
        # f2.show()
        save_path = folder_path + '/episode_' + str(episode_number)
        f2.set_size_inches(20, 14)
        f2.savefig(save_path)
        f2.clf()


def modify_pickle(folder_path, from_pickle, params):

    print('observation dim', from_pickle['observation_space'])
    print(from_pickle['stacked_frames'])
    # from_pickle['number_layers'] = 2
    # from_pickle['size_layers'] = 128
    #
    # # from_pickle['infos'] = infos
    # with open(folder_path + 'infos.pickle', 'wb') as file:
    #     pickle.dump(from_pickle, file)
    #     print('pickle saved')


def action_on_pickle(folder_path, count, el):
    if os.path.exists(folder_path):
        print(folder_path)
        count[0] += 1
        # print(count)
        # print(el)
        # if el != '0':
        #     print('hi')
        #     os.rename(old_name, new_name)
        with open(folder_path + 'infos.pickle', 'rb') as file:
            from_pickle = pickle.load(file)
        save_reward(folder_path, from_pickle)
        save_trajectory(folder_path, from_pickle)
        plot_learning_graphs(folder_path, from_pickle)
        # modify_pickle(folder_path, from_pickle, el)


i = [0]
if mode_loop:
    for str_N in number_agents:
        for obs in observation:
            for alp in learning_rate:
                for t in tau:
                    for k in decay_tau_factor:
                        for gam in gamma:
                            for name in test_names:
                                folder = './Saved_results/number_agents_' + str_N
                                folder += '/observation_' + obs
                                folder += '/gamma_' + gam
                                folder += '/2_layers_size_128'
                                folder += '/learning_rate_10e' + alp + '_tau_10e' + t
                                # new = folder
                                folder += '_decay_tau_' + k
                                # new += '_decay_tau_0'
                                # old = folder
                                folder += '/' + name + '/'
                                # print(folder)

                                action_on_pickle(folder, i, k)

else:
    for str_N, obs, alp, t, k, gam, name in zip(number_agents, observation, learning_rate, tau, decay_tau_factor, gamma,
                                                test_names):
        # print(N, obs, alp, gam)
        # print(type(N), type(obs), type(alp), type(gam))
        folder = './Saved_results/number_agents_' + str_N
        folder += '/observation_' + obs
        folder += '/gamma_' + gam
        folder += '/learning_rate_10e' + alp + '_tau_10e' + t
        folder += '_decay_tau_' + k
        folder += '/' + name + '/'
        action_on_pickle(folder, i, k)

print('number simulations = ', i[0])
