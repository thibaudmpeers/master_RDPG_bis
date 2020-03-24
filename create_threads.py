import os
from threading import Thread


class Task(Thread):
    def __init__(self, command_line):
        Thread.__init__(self)
        self.command_line = command_line

    def run(self):
        os.system(self.command_line)
        # print(self.command_line)


number_steps = 1000
number_episode = 2001
mode_action = 1
type_action = [True, False]
stacked_frames = False

available_obs = [0, 1, 2]
# available_obs = [1]
number_agents = [6]
observation_front = [True]
# observation_front = [True]
# alpha = [1e-7]
# alpha = [1e-4, 1e-5, 1e-3, 1e-2]
alpha = [1e-6]
tau = [1e-1, 1e-2]
decay_tau = [0]

threads = []

for n in number_agents:
    for alp in alpha:
        for t in tau:
            for k in decay_tau:
                for obs in available_obs:
                    for obs_front in observation_front:
                        if not obs_front and obs != 0:
                            continue
                        command = 'python3 main.py -s ' + str(number_steps)
                        command += ' -e ' + str(number_episode)
                        command += ' -a_o ' + str(obs)
                        command += ' -o_f' if obs_front and obs == 0 else ''
                        if mode_action == 0:
                            command += ' -a_ref'
                        elif mode_action == 1:
                            command += ' -a_u'
                            command += ' -e_p' if type_action[0] else ''
                            command += ' -e_i' if type_action[1] else ''
                        else:
                            command += ' -a_k'
                            command += ' -e_p' if type_action[0] else ''
                            command += ' -e_i' if type_action[1] else ''
                            command += ' -e_d' if type_action[2] else ''
                        command += ' -n ' + str(n)
                        command += ' -sf' if stacked_frames else ''
                        command += ' -alpha ' + str(alp)
                        command += ' -tau ' + str(t)
                        command += ' -k_tau ' + str(k)
                        threads.append(Task(command))

max_threads = 3

number_experiments = len(threads)

print('number of experiments: ', number_experiments)


for j in range(number_experiments):
    threads[j].start()
    if j % max_threads == max_threads - 1:
        for i in range(max_threads):
            threads[j-i].join()
    elif j == number_experiments - 1:
        for i in range(number_experiments % max_threads):
            threads[j - i].join()

plot_thread = Task('python3 pickle_analysis.py')
plot_thread.start()
plot_thread.join()