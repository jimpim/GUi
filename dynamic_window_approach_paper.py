"""
Dynamic Window Approach (DWA) as a module that can be stepped repeatedly.
"""

import math
import numpy as np

class Config:
    def __init__(self):
        self.max_speed = 1.0
        self.min_speed = 0.0
        self.max_yaw_rate = 40.0 * math.pi / 180.0
        self.max_accel = 0.2
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0
        self.v_resolution = 0.01
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0
        self.dt = 0.1
        self.predict_time = 1.0

        self.to_goal_cost_gain = 0.2
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 0.1

        self.goal_tolerance = 0.5

        # For stuck detection
        self.robot_stuck_flag_cons = 0.001

def motion(x, u, dt):
    """
    x: [x, y, yaw, v, omega]
    u: [v, omega]
    """
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_dynamic_window(x, config):
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]
    Vd = [
        x[3] - config.max_accel * config.dt,
        x[3] + config.max_accel * config.dt,
        x[4] - config.max_delta_yaw_rate * config.dt,
        x[4] + config.max_delta_yaw_rate * config.dt
    ]
    return [
        max(Vs[0], Vd[0]),
        min(Vs[1], Vd[1]),
        max(Vs[2], Vd[2]),
        min(Vs[3], Vd[3])
    ]

def predict_trajectory(x_init, v, w, config):
    x = np.array(x_init)
    traj = [x.copy()]
    time = 0.0
    while time <= config.predict_time:
        x = motion(x, [v, w], config.dt)
        traj.append(x.copy())
        time += config.dt
    return np.array(traj)

def calc_to_goal_cost(traj, goal):
    dx = goal[0] - traj[-1, 0]
    dy = goal[1] - traj[-1, 1]
    angle_to_goal = math.atan2(dy, dx)
    heading_error = angle_to_goal - traj[-1, 2]
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
    return abs(heading_error)

def calc_obstacle_cost(traj, ob, robot_radius=0.5):
    # Simple: 1/min_dist
    min_dist = float("inf")
    for px, py, _, _, _ in traj:
        dists = np.hypot(ob[:,0] - px, ob[:,1] - py)
        local_min = np.min(dists)
        if local_min < min_dist:
            min_dist = local_min
    # If min_dist < radius, big cost
    safe_dist = max(min_dist - robot_radius, 1e-3)
    return 1.0 / safe_dist

def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    best_u = [0.0, 0.0]
    best_traj = None
    min_cost = float("inf")

    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for w in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            traj = predict_trajectory(x, v, w, config)
            to_goal_c = config.to_goal_cost_gain * calc_to_goal_cost(traj, goal)
            speed_c   = config.speed_cost_gain   * (config.max_speed - traj[-1,3])
            ob_c      = config.obstacle_cost_gain * calc_obstacle_cost(traj, ob)

            final_cost = to_goal_c + speed_c + ob_c
            if final_cost < min_cost:
                min_cost = final_cost
                best_u = [v, w]
                best_traj = traj

    # Stuck check
    if abs(best_u[0]) < config.robot_stuck_flag_cons and abs(x[3]) < config.robot_stuck_flag_cons:
        best_u[1] = -config.max_delta_yaw_rate

    return best_u, best_traj

class DWAHandler:
    """
    A handy class that stores robot state, obstacles, config,
    and allows you to call .step() each iteration.
    """
    def __init__(self, x_init, goal, ob):
        self.x = np.array(x_init)  # [x, y, yaw, v, omega]
        self.goal = np.array(goal)
        self.ob = np.array(ob)  # Nx2
        self.config = Config()

        self.trajectory = [self.x[:2].copy()]

    def step(self):
        """
        Perform one iteration of DWA, update self.x
        returns: (x, is_goal_reached)
        """
        u, traj = dwa_control(self.x, self.config, self.goal, self.ob)
        self.x = motion(self.x, u, self.config.dt)
        self.trajectory.append([self.x[0], self.x[1]])

        dist_to_goal = math.hypot(self.x[0] - self.goal[0], self.x[1] - self.goal[1])
        is_goal = dist_to_goal <= self.config.goal_tolerance
        return self.x, is_goal

# No "main" so we can import this file without auto-running.
