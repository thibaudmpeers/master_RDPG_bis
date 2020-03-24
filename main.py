from __future__ import division
import numpy as np
import torch
import os
import sys
import gc
import matplotlib.pyplot as plt
import pickle
import argparse
import time
import DDPG.buffer as buffer
import DDPG.train as train
from DDPG.train import get_nn_arch_params
from Environment.LoopFollowingAgents import LoopFollowingAgents
sys.path.append(os.getcwd())

t_init = time.time()

E_pickle = 50

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--nb_steps', type=int, required=True)
parser.add_argument('-e', '--nb_episodes', type=int, required=True)
parser.add_argument('-a_o', '--available_obs', type=int)
parser.add_argument('-o_ui', '--obs_int_term', action='store_true')
parser.add_argument('-a_ref', '--action_on_reference', action='store_true')
parser.add_argument('-a_u', '--action_on_terms', action='store_true')
parser.add_argument('-a_k', '--action_on_gains', action="store_true")
parser.add_argument('-e_p', '--enable_action_P', action='store_true')
parser.add_argument('-e_d', '--enable_action_D', action='store_true')
parser.add_argument('-e_i', '--enable_action_I', action='store_true')
parser.add_argument('-n', '--nb_agents', type=int)
parser.add_argument('-rs', '--random_start', type=int)
parser.add_argument("-gr", '--gains_reward', type=float, nargs=4)
parser.add_argument('-ui', '--enable_integral_term', action='store_true')
parser.add_argument('-gpu', '--enable_gpu', action='store_true')
parser.add_argument('-p', '--plot_sum_reward', action='store_true')
parser.add_argument('-ex', '--only_exploitation', action='store_true')
parser.add_argument('-sf', '--stacked_frames', action='store_true')
parser.add_argument('-o_f', '--obs_front_agent_u', action='store_true')
parser.add_argument('-act_ag', '--active_agents', type=int, nargs='*')
parser.add_argument('-alpha', '--learning_rate', type=float)
parser.add_argument('-tau', '--tau_ddpg', type=float)
parser.add_argument('-k_tau', '--decay_tau', type=float)

args = parser.parse_args()

number_steps = args.nb_steps

number_episodes = args.nb_episodes

available_obs = args.available_obs

obs_front = True if available_obs == 0 and args.obs_front_agent_u else False

obs_u_i = args.obs_int_term

action_P = args.enable_action_P
action_D = args.enable_action_D
action_I = args.enable_action_I

choice = False
if args.action_on_reference:
    mode_action = 0
    choice = True
    type_action = None
    str_action = '/action_on_reference'

if args.action_on_terms:
    if choice:
        print('Multiple action modes have been chosen')
        sys.exit()
    mode_action = 1
    choice = True
    type_action = [True if (action_P or action_D) else False, action_I]
    str_action = '/action_on_terms_' + str(type_action).replace(' ', '')

if args.action_on_gains:
    if choice:
        print('Multiple action modes have been chosen')
        sys.exit()
    mode_action = 2
    choice = True
    type_action = [action_P, action_I, action_D]
    str_action = 'action_on_gains_' + str(type_action).replace(' ', '')

if not choice:
    print('No action mode selected')
    sys.exit()

print('mode action = ', mode_action)
print('type action = ', type_action)

nb_agents = args.nb_agents if args.nb_agents else 6

random_start = args.random_start if args.random_start else 1

gains = args.gains_reward if args.gains_reward else [1, 1, 0.25, 10]

u_i = args.enable_integral_term

plot_sum_reward = args.plot_sum_reward

only_exploitation = args.only_exploitation

stacked_frames = args.stacked_frames

device = torch.device("cuda") if args.enable_gpu else torch.device("cpu")

active_agents = args.active_agents if args.active_agents else [0]

learning_rate = args.learning_rate if args.learning_rate else 1e-5

tau = args.tau_ddpg if args.tau_ddpg else 1e-5

decay_tau = args.decay_tau if args.decay_tau else 0

print(active_agents)
delay = 0.1
time_step = 0.025
local_time_step = 0.0125
dist_ref = 2


env = LoopFollowingAgents(available_obs=available_obs, obs_u_i=obs_u_i, mode_action=mode_action,
                          type_action=type_action, number_agents=nb_agents, random_start=random_start, delay=delay,
                          dt=time_step, d_ref=dist_ref, local_time_step=local_time_step, active_agents=active_agents,
                          gains_reward=gains, u_i=u_i, stacked_frames=stacked_frames, obs_front=obs_front,
                          number_steps=number_steps)

size_replay_memory = 100

dim_obs = env.observation_space
dim_action = env.action_space

memory = buffer.MemoryBuffer(size_replay_memory, number_steps, dim_obs, dim_action)
manager_nn = train.Trainer(dim_obs, dim_action, number_steps, memory, device, learning_rate, tau, decay_tau)
learning_params = manager_nn.return_params()
random_params = manager_nn.noise.return_params()
number_layers, size_layers = get_nn_arch_params()

learning_rate = learning_params[1]
gamma = learning_params[2]
init_tau = learning_params[3]
decay_factor_tau = learning_params[4]

nb_actor_params, nb_critic_params = manager_nn.get_models_params()
print('actor parameters = ', nb_actor_params, '; critic_parameters = ', nb_critic_params)

path = './Saved_results/number_agents_' + str(nb_agents)  # + str_action + '/gains_' + str(gains).replace(' ', '')
# path += '/stacked_frames' if stacked_frames else '/no_stacked_frames'
path += '/observation_' + str(available_obs)
path += '+front_agent' if obs_front else ''
path += '/gamma_' + str(gamma)
# path += '/Exploitation/' if only_exploitation else '/Exploration/'
path += '/' + str(number_layers) + '_layers_size_' + str(size_layers) + '/'
path += '/learning_rate_10e' + str(int(np.log10(learning_rate))) + '_tau_10e' + str(int(np.log10(init_tau)))
path += '_decay_tau_0' if decay_factor_tau == 0 else '_decay_tau_10e' + str(int(np.log10(decay_factor_tau)))
path += '/test'

nb_tests = 1
while os.path.exists(path + str(nb_tests)):
    nb_tests += 1

path += str(nb_tests) + '/'
os.makedirs(path + 'Models/')

# S_DIM = env.observation_space.shape[0]
# A_DIM = env.action_space.shape[0]
# A_MAX = env.action_space.high[0]
to_pickle = {'number_agents': nb_agents, 'delay': delay, 'time_step': time_step, 'local_time_step': local_time_step,
             'distance_ref': dist_ref, 'active_agents': active_agents, 'number_layers': number_layers,
             'available_observations': available_obs, 'observation_integral_term': obs_u_i, 'mode_action': mode_action,
             'type_action': type_action, 'random_start': random_start, 'gains_reward': gains, 'enable_u_i': u_i,
             'stacked_frames': stacked_frames, 'number_episodes': number_episodes, 'number_steps': number_steps,
             'size_replay_memory': size_replay_memory, 'observation_space': dim_obs, 'action_space': dim_action,
             'learning_parameters': learning_params, 'random_parameters': random_params, 'size_layers': size_layers,
             'nb_actor_parameters': nb_actor_params, 'nb_critic_parameters': nb_critic_params, 'obs_front': obs_front,
             'feedback_parameters': env.feedback_params(), 'infos': [], 'saved_episode_number': [],
             'sum_reward_history': [], 'initial_states': [], 'critic_loss': [], 'actor_loss': []}

print(' State Dimensions :- ', dim_obs)
print(' Action Dimensions :- ', dim_action)

if plot_sum_reward:
    xdata = []
    ydata = []
    axes = plt.gca()
    line, = axes.plot(xdata, ydata, 'r-')

execution_times = np.zeros(number_episodes)

for episode in range(number_episodes):
    t_start = time.time()
    save_traj = True if episode % E_pickle == 0 else False

    memory.new_epsisode()

    observation, initial_state = env.reset(get_init_state=True, save_traj=save_traj)
    to_pickle['initial_states'].append(initial_state)
    sum_reward = 0

    manager_nn.update_tau(episode)

    previous_action = torch.zeros(1, dim_action, dtype=torch.float32)
    previous_reward = torch.zeros(1, 1, dtype=torch.float32)

    obs = torch.from_numpy(np.float32(observation).reshape(1, dim_obs))
    inputs = torch.cat((obs, previous_action, previous_reward), dim=1)

    h_target_critic = manager_nn.target_critic.init_hidden_state(1)
    h_actor = manager_nn.actor.init_hidden_state(1)
    h_target_actor = manager_nn.target_actor.init_hidden_state(1)

    target_action, h_target_actor = manager_nn.target_actor(inputs, h_target_actor)
    _, h_target_critic = manager_nn.target_critic(inputs, target_action, h_target_critic)

    # manager_nn.get_learning_rate()
    # print('EPISODE :- ', episode)
    for step in range(number_steps):

        if episode == number_episodes - 1 or episode % 10 == 0 or only_exploitation:
            action, h_actor = manager_nn.get_exploitation_action(inputs, h_actor)
        else:
            action, h_actor = manager_nn.get_exploration_action(inputs, h_actor)

        action_cpu = action.squeeze(0).cpu().numpy()

        new_observation, reward, done, info = env.step(action_cpu, step)
        # env.render()
        sum_reward += reward

        reward = torch.from_numpy(reward.astype(np.float32).reshape(1, 1))

        if done:
            new_state = None
        else:
            new_state = np.float32(new_observation)
            # push this exp in ram
            memory.save_trasition(obs, action, reward, inputs, step)

        obs = torch.from_numpy(np.float32(new_observation).reshape(1, dim_obs))
        previous_action = action
        previous_reward = reward

        inputs = torch.cat((obs, previous_action, previous_reward), dim=1)

        target_action, h_target_actor = manager_nn.target_actor(inputs, h_target_actor)
        target_value, h_target_critic = manager_nn.target_critic(inputs, target_action, h_target_critic)
        memory.save_target_value(target_value, step)

        if done:
            break

    critic_loss, actor_loss, mean = manager_nn.optimize()

    print('Episode ', episode, ': sum rewards = ', sum_reward)
    print('critic_loss = ', critic_loss)
    print('actor loss = ', actor_loss)
    print('mean values = ', mean)
    # check memory consumption and clear memory
    gc.collect()
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)
    t_stop = time.time()
    execution_times[episode] = t_stop - t_start

    to_pickle['sum_reward_history'].append(sum_reward)
    to_pickle['critic_loss'].append(critic_loss)
    to_pickle['actor_loss'].append(actor_loss)
    if episode % E_pickle == 0:
        to_pickle['infos'].append(info)
        to_pickle['saved_episode_number'].append(episode)
    if episode % 1000 == 0:
        with open(path + 'infos.pickle', 'wb') as file:
            pickle.dump(to_pickle, file)
    if episode % 500 == 0:
        manager_nn.save_models(episode, path)
    if plot_sum_reward:
        xdata.append(episode + 1)
        ydata.append(sum_reward)
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        plt.draw()
        axes.relim()
        axes.autoscale_view()
        plt.pause(0.0001)

to_pickle['execution_times'] = execution_times
with open(path + 'infos.pickle', 'wb') as file:
    pickle.dump(to_pickle, file)

total_time = time.time() - t_init

mean = sum(execution_times) / len(execution_times)
sd = np.sqrt(sum((execution_times - mean) ** 2) / len(execution_times))
print('mean execution time = ', mean)
print('standard deviation execution time = ', sd)
print('total time = ', total_time)
print('Completed episodes')
# plt.plot(execution_times)
# plt.show()
