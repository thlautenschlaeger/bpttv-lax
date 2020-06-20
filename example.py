import gym
from quanser_robots import GentlyTerminating

from lax.a2c_lax import learn

if __name__ == '__main__':
    seed = 42
    env = gym.make('Pendulum-v0')
    # env = GentlyTerminating(gym.make('CartpoleStabShort-v0'))
    # env = GentlyTerminating(gym.make('Qube-100-v0'))
    # env = GentlyTerminating(gym.make('CartpoleSwingShort-v0'))
    # env = GentlyTerminating(gym.make('LunarLanderContinuous-v2'))
    # env = GentlyTerminating(gym.make('BipedalWalker-v2'))
    # env = GentlyTerminating(gym.make('BipedalWalkerHardcore-v2'))
    # env = GentlyTerminating(gym.make('HalfCheetah-v3'))

    learn(env, seed=seed, obfilter=True, tsteps_per_batch=2500, cv_opt_epochs=5, lax=True, gamma=0.97, lamb=0.99,
          animate=True)