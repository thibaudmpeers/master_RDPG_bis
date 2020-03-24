import numpy as np
from Environment.utils import Gaussian_noise, Ornstein_Uhlenbeck, Load_disturbance


class LocalAgent:
    def __init__(self, local_time_step, length_env, max_speed, steps_delay=0, steps_action=1, u_i=True,
                 perturbations=False, load_disturbance=False):
        self.max_acceleration = 2

        self.steps_delay = steps_delay
        self.local_dt = local_time_step
        self.length_env = length_env
        self.steps_action = steps_action
        self.max_speed = max_speed

        self.perturbations = perturbations
        self.gauss = True
        self.noise = Gaussian_noise(1) if self.gauss else Ornstein_Uhlenbeck(1)

        # gain computation
        period_zero = 3
        omega_zero = 2*np.pi / period_zero
        self.kp = omega_zero**2
        self.kd = 2*omega_zero
        # self.kp = 6
        # self.kd = 2
        self.ki = self.kp/4 if u_i else 0
        self.speed_damping = 1/4

        self.previous_error = None
        self.saved_u_tot = None
        self.position = None
        self.speed = None
        self.u_i = None

        self.load_disturbance = load_disturbance
        if self.load_disturbance:
            max_load = 10
            nb_steps_load = int(0.5/self.local_dt)
            self.current_step = 0
            self.step_load = 1000
            self.load_fun = Load_disturbance(nb_steps=nb_steps_load, max_load=max_load)

    def take_step(self, error_position, error_speed, actions_u=None, actions_gains=None, enable_PID=True):
        actions_gains = [0, 0, 0] if actions_gains is None else actions_gains
        actions_u = [0, 0] if actions_u is None else actions_u

        u_p = (self.kp + actions_gains[0]) * error_position
        self.u_i += (self.ki + actions_gains[1]) * (error_position + actions_u[1]) * self.local_dt * self.steps_action
        u_d = (self.kd + actions_gains[2]) * error_speed

        u = self.u_i + u_p + u_d + actions_u[0] if enable_PID else actions_u[0]

        self.previous_error = error_position

        u_sat = np.clip(u, -self.max_acceleration, self.max_acceleration)

        self.u_i += self.ki * (u_sat - u) * self.local_dt * self.steps_action

        self.saved_u_tot = [u_sat] + self.saved_u_tot[:-1]
        delayed_u_tot = self.saved_u_tot[-1]
        if self.perturbations:
            delayed_u_tot += self.noise.sample()

        for _ in range(self.steps_action):
            self.position += self.local_dt * self.speed
            self.position = self.position % self.length_env

            self.speed += self.local_dt * (delayed_u_tot - self.speed_damping * self.speed)
            if self.load_disturbance:
                start = True if self.current_step == self.step_load else False
                self.speed += self.local_dt * self.load_fun.sample(start)

            self.speed = np.clip(self.speed, -self.max_speed, self.max_speed)
            self.current_step += 1

        return self.position, self.speed, u_sat, self.u_i

    def reset(self, init_position, init_error, init_speed=0):
        self.speed = init_speed
        self.position = init_position
        self.saved_u_tot = [0]*self.steps_delay
        self.previous_error = init_error
        self.u_i = 0
        self.current_step = 0

    def feedback_params(self):
        return [self.kp, self.ki, self.kd]
