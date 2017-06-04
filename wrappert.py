import gym
import types


def copy(self):
    return gym.make(self.spec.id)


env = gym.make('CartPole-v0')
env.copy = types.MethodType(copy, env)
