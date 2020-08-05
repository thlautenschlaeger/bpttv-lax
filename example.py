import gym
from quanser_robots import GentlyTerminating

from lax.a2c_lax import learn


if __name__ == '__main__':
    seed = 42
    # env = gym.make('Pendulum-v0')
    # env = GentlyTerminating(gym.make('CartpoleStabShort-v0'))
    # env = GentlyTerminating(gym.make('Qube-100-v0'))
    env = GentlyTerminating(gym.make('CartpoleSwingShort-v0'))
    # env = GentlyTerminating(gym.make('LunarLanderContinuous-v2'))
    # env = GentlyTerminating(gym.make('BipedalWalker-v2'))
    # env = GentlyTerminating(gym.make('BipedalWalkerHardcore-v2'))
    # env = GentlyTerminating(gym.make('HalfCheetah-v3'))

    # env.unwrapped._dt = 0.01
    # env.unwrapped._sigma = 1e-4
    # env.spec._max_episode_steps = 100
    # env._max_episode_steps = 100

    learn(env, seed=seed, obfilter=True, total_steps=int(50e6), tsteps_per_batch=5000, cv_opt_epochs=5, lax=False,
          gamma=0.99, lamb=0.97, check_kl=True, animate=True, vf_opt_epochs=50, save_loc='evals')
