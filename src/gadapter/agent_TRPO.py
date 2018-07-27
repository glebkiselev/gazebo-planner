from itertools import count
from collections import OrderedDict
import numpy as np
import rospy

from gazeboplanner.gadapter.util import *
from gazeboplanner.gadapter.network import Network

class TRPOAgent:
    n_actions = 4  # type: int
    observation_shape = (8,)  # type: tuple

    def __init__(self, env):
        self.env = env
        self.init_network()
        self.n_actions = 4
        self.observation_shape = (8,)

    def act(self, obs, sample=True):
        m = self.net.get_mean([obs])[0]
        log_std = self.net.get_logstd([obs])[0]
        rnd = np.random.normal(size=self.n_actions)
        if sample:
            action = np.exp(log_std) * rnd + m
        else:
            action = m
        return action, m, log_std

    def init_network(self):
        self.net = Network(self.n_actions, self.observation_shape)


    def train_path(self, max_kl=0.01, cg_damping=0.1, numeptotal=0, max_pathlength=500, n_timesteps=30000):
        print("Rollout")
        paths = rollout(self.env, self, max_pathlength, n_timesteps)
        print("Made rollout")

        # Updating policy.
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        returns = np.concatenate([path["cumulative_returns"] for path in paths])
        old_m = np.concatenate([path["m"] for path in paths])
        old_logstd = np.concatenate([path["logstd"] for path in paths])
        inputs_batch = [observations, actions, returns, old_m, old_logstd]

        old_weights = self.net.get_flat_weights()

        def fisher_vector_product(p):
            """gets intermediate grads (p) and computes fisher*vector """
            return self.net.compute_fisher_vector_product(observations, p) + cg_damping * p

        flat_grad = self.net.get_surrogate_gradients(*inputs_batch)

        stepdir = conjugate_gradient(fisher_vector_product, -flat_grad)
        shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
        lm = np.sqrt(shs / max_kl)
        fullstep = stepdir / lm

        # Compute new weights with linesearch in the direction we found with CG

        def losses_f(flat_weights):
            self.net.load_flat_weights(flat_weights)
            return self.net.compute_losses(*inputs_batch)

        new_weights = linesearch(losses_f, old_weights, fullstep, max_kl)

        self.net.load_flat_weights(new_weights)

        # Report current progress
        L_surr, kl, entropy = self.net.compute_losses(*inputs_batch)
        episode_rewards = np.array([path["rewards"].sum() for path in paths])

        stats = OrderedDict()
        numeptotal += len(episode_rewards)
        stats["Total number of episodes"] = numeptotal
        stats["Average sum of rewards per episode"] = episode_rewards.mean()
        stats["Std of rewards per episode"] = episode_rewards.std()
        stats["Entropy"] = entropy
        stats["KL between old and new distribution"] = kl
        stats["Surrogate loss"] = L_surr
        return stats, episode_rewards

    def learn(self, reward=530,  max_pathlength=500, n_timesteps=30000):
        rewards = []
        for i in count(1):
            print("\n********** Iteration %i ************" % i)
            stats, episode_rewards = self.train_path(max_pathlength=max_pathlength, n_timesteps=n_timesteps)
            if episode_rewards.mean() > reward:
                break
            rewards.append(episode_rewards.mean())
            for k, v in stats.items():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            i += 1
        return rewards, i

    def play(self, env):
        obs = env.reset()
        done = False
        reward = 0
        l = 0
        while done is not True:
            a = self.act(obs, sample=False)[0]
            for i in range(self.n_actions):
                obs, r, done = env.step([i, a[i]])
                if done:
                    break
            reward += r
            l += 1
            print(l, ')', r, obs, a)

    def grasp(self, env):
        if env:
            print(True)
        obs = synthetic_state(env, env.render(mode = 'human'), env.aim)
        done = False
        reward = 0
        l = 0
        while done is not True:
            env.gripper.publish(2.0)
            a = self.act(obs, sample=False)[0]
            for i in range(self.n_actions):
                obs, r, done = env.step([i, a[i]])
                obs = synthetic_state(env, obs, env.aim)
                if done:
                    break
            reward += r
            l += 1
            print(l, ')', r, obs, a)
        env.gripper.publish(0.0)
        rospy.sleep(10.0)

    def pick(self, env):
        env.gripper.publish(0.0)
        env.h += 0.1
        obs = synthetic_state(env, env.render(mode = 'human'), env.aim)

        done = False
        reward = 0
        l = 0
        while done is not True:
            a = self.act(obs, sample=False)[0]
            for i in range(self.n_actions):
                obs, r, done = env.step([i, a[i]])
                obs = synthetic_state(env, obs, env.aim)
                if done:
                    break
            reward += r
            l += 1
            print(l, ')', r, obs, a)

    def putdown(self, env):
        obs = synthetic_state(env, env.render(mode = 'human'), env.aim)
        env.h += -0.1
        done = False
        reward = 0
        l = 0
        while done is not True:
            a = self.act(obs, sample=False)[0]
            for i in range(self.n_actions):
                obs, r, done = env.step([i, a[i]])
                obs = synthetic_state(env, obs, env.aim)
                if done:
                    break
            reward += r
            l += 1
            print(l, ')', r, obs, a)
