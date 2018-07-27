import numpy as np
import theano.tensor as T
from theano.gradient import disconnected_grad as const
from math import radians
import scipy.signal
from gym_crumb.envs.crumb_pick_env import metric


def kl_sym(old_mean, new_mean, old_logstd, new_logstd):
    numerator = (old_mean - new_mean) ** 2 + T.exp(2 * old_logstd) - T.exp(2 * new_logstd)
    denominator = 2 * T.exp(2 * new_logstd) + 1e-8
    return T.mean(T.sum((numerator / denominator + 0.5 * new_logstd - 0.5 * old_logstd), axis=-1))


def kl_sym_firstfixed(mean, logstd):
    numerator_firstfixed = (const(mean) - mean) ** 2 + const(T.exp(logstd)) - T.exp(logstd)
    denominator_firstfixed = 2 * T.exp(logstd) + 1e-8
    return T.mean(T.sum((numerator_firstfixed / denominator_firstfixed + 0.5 * logstd - 0.5 * const(logstd)), axis=-1))


def linesearch(f, x, full_step, max_kl):
    max_backtracks = 10
    loss, _, _ = f(x)
    for stepfrac in .5 ** np.arange(max_backtracks):
        x_new = x + stepfrac * full_step
        new_loss, kl, _ = f(x_new)
        actual_improve = new_loss - loss
        if kl <= max_kl and actual_improve < 0:
            x = x_new
            loss = new_loss
    return x


def get_flat_gradient(loss, var_list):
    grads = T.grad(loss, var_list)
    return T.concatenate([grad.ravel() for grad in grads])


def slice_vector(vector, shapes):
    start = 0
    tensors = []
    for shape in shapes:
        size = T.prod(shape)
        tensor = vector[start:(start + size)].reshape(shape)
        tensors.append(tensor)
        start += size
    return tensors


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (p.dot(z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def get_cummulative_returns(r, gamma=0.9):
    r = np.array(r)
    assert r.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], r[::-1], axis=0)[::-1]


def rollout(env, agent, max_pathlength=2500, n_timesteps=50000):
    paths = []
    total_timesteps = 0
    env.state = np.zeros(4)
    while total_timesteps < n_timesteps:
        observations, actions, rewards, action_m, action_logstd = [], [], [], [], []
        observation = env.reset()
        done = False
        for _ in range(max_pathlength):
            action, m, logstd = agent.act(observation)
            observations.append(observation)
            actions.append(action)
            action_m.append(m)
            action_logstd.append(logstd)
            reward = 0
            for i in range(agent.n_actions):
                observation, r, done = env.step([i, action[i]])
                reward += r
            rewards.append(reward)
            total_timesteps += 1
            if done or total_timesteps%max_pathlength == 0:
                path = {"observations": np.array(observations),
                        "m": np.array(action_m),
                        "logstd": np.array(action_logstd),
                        "actions": np.array(actions),
                        "rewards": np.array(rewards),
                        "cumulative_returns": get_cummulative_returns(rewards),
                        }
                paths.append(path)
                break
    return paths


def log_probability(logstd, mean, actions):
    return - T.sum(logstd, -1) - 0.5 * T.sum(T.square((actions - mean) / T.exp(logstd)), -1) - 0.5 * np.log(2 * np.pi)


#-------------------------------------------------------------
sign = lambda a: 1 if a > 0 else -1 if (a < 0) else 0


def signs(vec_a):
    s = np.zeros(len(vec_a))
    for i in range(len(vec_a)):
        s[i] = sign(vec_a[i])
    return s


def module(vec_a):
    return ((vec_a ** 2).sum(axis=1) ** 0.5).reshape(3, 1)


def angle(vec_a, vec_b):
    return np.arccos(((vec_a / module(vec_a)) * (vec_b / module(vec_b))).sum(axis=1))


def norm(vec_a):
    n = np.zeros_like(vec_a)
    for i in range(len(vec_a)):
        n[i][0] = -vec_a[i][1]
        n[i][1] = vec_a[i][0]
    return n


def get_pose(joint, env):
    a = env.link_state(joint, 'base_footprint').link_state.pose.position
    return np.array([a.x, a.y, a.z])


def state1(aim, env):
    aim1 = np.array([aim.x, aim.y, aim.z])
    # Joints pose
    joint = np.zeros((4, 3))
    joint[0] = get_pose('biceps_link', env)
    joint[1] = get_pose('forearm_link', env)
    joint[2] = get_pose('wrist_1_link', env)
    joint[3] = get_pose('gripper_1_link', env)*0.5 + get_pose('gripper_2_link', env)*0.5

    joint[3] += (joint[3] - joint[2])*(0.025 / 0.11450014)
    # vectors
    vec1 = np.zeros(3)
    vec2 = np.zeros(3)

    for i in range(3):
        vec1 = joint[3] - joint[0]
        vec2 = aim1 - joint[0]

    #state = angle(norm(vec1), vec2) // 0.01 * 0.01
    vec = tuple((vec1 - vec2)//0.01*0.01)
    return vec# + (state[0], state[1], state[2])



def synthetic_state(env, s, aim):
    s = tuple([s[i] // 0.01 * 0.01 for i in range(0, 4)])
    gripper = env.link_state('gripper_1_link', 'base_footprint').link_state.pose.position
    r = metric(gripper, env.aim) // 0.01 * 0.01
    state = (r,) + state1(env.aim, env) + s
    return state