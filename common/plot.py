import torch
import matplotlib.pyplot as plt


def plt_expected_cum_reward(path, expected_reward, std):
    """
    path to store evaluation plots

    :param expected_reward:
    :return:
    """

    fig, ax = plt.subplots()
    ax.plot(expected_reward, color='red')
    ax.fill_between(list(range(len(expected_reward))), torch.tensor(expected_reward) -
                    torch.tensor(std), torch.tensor(expected_reward) + torch.tensor(std), alpha=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Expected Cumulative Reward')
    ax.grid(alpha=0.5, linestyle='-')


    fig.savefig(path+'/reward_lax_True_cv_mins_2500_act_detach-25opt_bipedal-hardcore-larger-net-kl-adapted.png')
    plt.clf()
    plt.close('all')
