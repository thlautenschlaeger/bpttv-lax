import torch
import gym
import numpy as np
from copy import deepcopy

from quanser_robots import GentlyTerminating

from common.filters import ZFilter
from common.plot import plt_expected_cum_reward
from lax.policies import LaxPolicyModel
from common import calc
from common.util import set_global_seeds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()


class Model(object):

    def __init__(self, obs_dim, act_dim, cv_epochs, vf_epochs):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cv_epochs = cv_epochs
        self.vf_epochs = vf_epochs
        self.train_policy_model = LaxPolicyModel(obs_dim, act_dim, p_lr=1e-4)
        self.step_policy_model = deepcopy(self.train_policy_model)
        # self.step_policy_model = LaxPolicyModel(obs_dim, act_dim, p_lr=1e-3)

    def get_cv_grads(self, obs, actions, rewards, vf_in, values, net_mean, net_std):
        advs = rewards - values

        pg_grads = self.get_policy_grads(obs, actions, rewards, vf_in, values, net_mean, net_std)

        cv_grad = torch.cat([torch.reshape(p, [-1]) for p in pg_grads], 0)
        cv_loss = torch.square(cv_grad).mean(0)

        cv_grad = torch.autograd.grad(cv_loss, self.step_policy_model.cv_net.parameters(),
                                      grad_outputs=torch.ones_like(cv_loss))

        return cv_grad

    def update_cv(self, cv_grad):
        self.step_policy_model.cv_optim.zero_grad()
        for params, cg in zip(self.step_policy_model.cv_net.parameters(), cv_grad):
            params.backward(cg, retain_graph=True)
        self.step_policy_model.cv_optim.step()

    def get_policy_grads(self, obs, actions, rewards, vf_in, value, net_mean, net_std):
        advs = rewards - value

        # net_mean, net_std = self.train_policy_model.policy(obs)
        # actions = torch.distributions.Normal(net_mean, net_std).rsample()
        cv = self.step_policy_model.cv_net(torch.cat([vf_in, actions], dim=1))
        ddiff_loss = torch.mean(cv, dim=0)

        dist = torch.distributions.Normal(net_mean, net_std)

        log_prob = dist.log_prob(actions.detach())
        entropy = dist.entropy()

        dlog_prob = -((advs[:, None] - cv) * log_prob).mean(dim=0) - entropy.mean(0) * 0.01

        pg_grads_mean = torch.autograd.grad(dlog_prob, self.step_policy_model.policy.net.parameters(),
                                            grad_outputs=torch.ones_like(dlog_prob),
                                            create_graph=True, retain_graph=True)
        pg_grads_std = torch.autograd.grad(dlog_prob, self.step_policy_model.policy.log_std,
                                           grad_outputs=torch.ones_like(dlog_prob),
                                           create_graph=True, retain_graph=True)

        ddiff_grads_mean = torch.autograd.grad(ddiff_loss, self.step_policy_model.policy.net.parameters(),
                                               grad_outputs=torch.ones_like(ddiff_loss),
                                               create_graph=True, retain_graph=True)
        ddiff_grads_std = torch.autograd.grad(ddiff_loss, self.step_policy_model.policy.log_std,
                                              grad_outputs=torch.ones_like(ddiff_loss),
                                              create_graph=True, retain_graph=True)

        pg_grads_mean = [pg - dg for pg, dg in zip(pg_grads_mean, ddiff_grads_mean)]
        pg_grads_std = [pg - dg for pg, dg in zip(pg_grads_std, ddiff_grads_std)]

        pg_grads = pg_grads_std + pg_grads_mean

        return pg_grads

    def update_vf(self, vf_in, rewards):
        t_loss = 0
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.vf_epochs):
            pred_reward = self.train_policy_model.vf_net(vf_in)
            # loss = (torch.square(rewards[:, None] - pred_reward)).mean(0)
            loss = loss_fn(rewards[:, None], pred_reward)
            self.train_policy_model.vf_optim.zero_grad()
            loss.backward()
            self.train_policy_model.vf_optim.step()
            t_loss += loss.detach().cpu().numpy()
        return t_loss / self.vf_epochs

    def update_policy(self, pg):
        self.train_policy_model.cv_optim.zero_grad()
        for params, _pg in zip(self.train_policy_model.policy.parameters(), pg):
            params.backward(_pg, retain_graph=True)
        self.train_policy_model.policy_optim.step()


class RolloutRunner(object):

    def __init__(self, env: gym.Env, policy_model: LaxPolicyModel, max_path_len, gamma=0.99, lamb=0.97, obfilter=None,
                 animate=False):
        self.env = env
        self.step_policy_model = policy_model
        self.max_path_len = max_path_len
        self.gamma = gamma
        self.lamb = lamb
        self._num_rollouts = 0
        self._num_steps = 0
        self.animate = animate

        self.obfilter = obfilter

    def run(self):
        ob = self.env.reset()
        if self.obfilter:
            ob = self.obfilter(ob)
        ob = to_torch(ob)
        prev_ob = torch.zeros_like(ob)

        done = False

        obs, acts, ac_dist_mean, ac_dist_std, logps, rewards = [], [], [], [], [], []

        for _ in range(self.max_path_len):
            state = torch.cat([ob, prev_ob], -1)
            obs.append(state)
            sampled_ac, log_prob, mean, std = self.step_policy_model.act(state)
            acts.append(sampled_ac)
            ac_dist_mean.append(mean)
            ac_dist_std.append(std)
            logps.append(log_prob)
            prev_ob = torch.clone(ob)
            scaled_act = self.env.action_space.low + (to_npy(sampled_ac) + 1.) * 0.5 * (self.env.action_space.high - self.env.action_space.low)
            scaled_act = np.clip(scaled_act, a_min=self.env.action_space.low, a_max=self.env.action_space.high)
            ob, reward, done, _ = self.env.step(scaled_act)
            if self.animate:
                self.env.render()

            if self.obfilter: ob = self.obfilter(ob.squeeze())
            ob = to_torch(ob)
            rewards.append(reward)

            if done:
                break

        self._num_rollouts += 1
        self._num_steps += len(rewards)

        path = {"observation": torch.stack(obs), "terminated": done,
                "reward": torch.tensor(rewards), "action": torch.stack(acts),
                "act_mean": torch.stack(ac_dist_mean), "act_std": torch.stack(ac_dist_std),
                "logp": torch.stack(logps)}

        rew_t = path["reward"]
        value = self.step_policy_model.predict_value(path).detach()
        vtarget = to_torch((calc.discount(np.append(to_npy(rew_t), 0.0 if path["terminated"] else value[-1]), self.gamma)[:-1]).copy())
        vpred_t = torch.cat([value, torch.zeros([1])]) if path["terminated"] else torch.cat([value, value[-1]])
        delta_t = rew_t + self.gamma * vpred_t[1:] - vpred_t[:-1]
        adv_GAE = to_torch(calc.discount(to_npy(delta_t), self.gamma * self.lamb).copy())

        return path, vtarget, value, adv_GAE


def learn(env: gym.Env, seed=None, total_steps=int(10e6),
          cv_opt_epochs=5, vf_opt_epochs=25, gamma=0.99, lamb=0.97, tsteps_per_batch=2500,
          kl_thresh=0.002, obfilter=True, lax=True, update_targ_interval=2, animate=False, save_loc=None):

    if seed:
        set_global_seeds(seed)
        env.seed(seed)

    obfilter = ZFilter(env.observation_space.shape) if obfilter else None

    max_pathlength = env.spec.max_episode_steps
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    model = Model(obs_dim, act_dim, cv_opt_epochs, vf_opt_epochs)
    runner = RolloutRunner(env, model.step_policy_model, max_pathlength, gamma=gamma, lamb=lamb, obfilter=obfilter,
                           animate=animate)

    tsteps_so_far = 0
    episode = 0
    best_exp_mean = -np.Inf
    exp_rews_means = []
    exp_rews_stds = []

    update_step_count = 0
    while True:
        if tsteps_so_far > total_steps:
            break
        print("Episode: {}".format(episode + 1))
        episode += 1

        tsteps_this_batch = 0
        paths = []
        vtargs = []
        advs = []
        std_advs = []
        vf_ins = []
        values = []
        cv_grads = []
        while True:
            path, vtarget, value, adv = runner.run()

            if lax:
                std_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                vf_in = model.step_policy_model.preproc(path)
                cv_grad = model.get_cv_grads(path["observation"], path["action"], vtarget, vf_in, value.detach(),
                                                   path["act_mean"], path["act_std"])
                cv_grads.append(cv_grad)
                std_advs.append(std_adv)
                vf_ins.append(vf_in)
            vtargs.append(vtarget)
            advs.append(adv)
            values.append(value)
            paths.append(path)
            n = path["reward"].shape[0]
            tsteps_this_batch += n
            tsteps_so_far += n

            if tsteps_this_batch > tsteps_per_batch:
                break

        ob_no = torch.cat([path["observation"] for path in paths])
        action_na = torch.cat([path["action"] for path in paths])
        oldac_mean = torch.cat([path["act_mean"] for path in paths])
        oldac_std = torch.cat([path["act_std"] for path in paths])
        log_probs_n = torch.cat([path["logp"] for path in paths])
        adv_n = torch.cat(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        rewards_n = torch.cat(vtargs)
        values_n = torch.cat(values)
        values_n = values_n.detach()

        all_rewards = torch.stack([path["reward"].sum() for path in paths])
        mean_reward = all_rewards.mean()
        std_reward = all_rewards.std()
        exp_rews_means.append(mean_reward)
        exp_rews_stds.append(std_reward)

        print("Expected reward: {} ± {}".format(mean_reward, std_reward))

        x = torch.cat([model.step_policy_model.preproc(p) for p in paths])
        x = x.detach()

        vf_loss = model.update_vf(x, rewards_n)
        if lax:
            pg = model.get_policy_grads(ob_no, action_na, rewards_n, x, values_n, oldac_mean, oldac_std)
            model.update_policy(pg)

            for r in range(cv_opt_epochs):
                cv_gs = []
                for k in range(len(cv_grads[0])):
                    cvg = 0
                    for l in range(len(cv_grads)):
                        cvg += cv_grads[l][k]
                    cvg /= len(cv_grads)
                    cv_gs.append(cvg)
                model.update_cv(cv_gs)

                cv_grads = []
                for p, _ in enumerate(paths):
                    cv_grads.append(
                        model.get_cv_grads(paths[p]["observation"], paths[p]["action"], vtargs[p], vf_ins[p], values[p].detach(),
                                           paths[p]["act_mean"], paths[p]["act_std"]))
        else:
            adv = rewards_n - values_n
            # action is detached
            log_probs_n = - torch.sum(torch.log(oldac_std), dim=1) - 0.5 * torch.log(
                torch.tensor(2.0 * np.pi)) * act_dim - 0.5 * torch.sum(
                torch.square(oldac_mean - action_na.detach()) / (torch.square(oldac_std)), dim=1)
            entropy = torch.distributions.Normal(oldac_mean, oldac_std).entropy()
            ploss = -(adv * log_probs_n).mean(0) - entropy.mean() * 0.01

            pg = torch.autograd.grad(ploss, model.step_policy_model.policy.parameters(),
                                     retain_graph=True, create_graph=True)
            model.train_policy_model.policy_optim.zero_grad()
            for modparams, grads in zip(model.train_policy_model.policy.parameters(), pg):
                modparams.backward(grads)
            model.train_policy_model.policy_optim.step()
            print("Policy loss: {} -|- vf loss: {}".format(ploss.detach().cpu().numpy(), vf_loss))


        kl = model.train_policy_model.compute_kl(ob_no, oldac_mean, oldac_std)

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)

        if kl > kl_thresh * 2:
            for g in model.train_policy_model.policy_optim.param_groups:
                g['lr'] = max(min_stepsize, g['lr'] / 1.5)
            print("KL too high # {}".format(kl))
        elif kl < kl_thresh / 2:
            for g in model.train_policy_model.policy_optim.param_groups:
                g['lr'] = min(max_stepsize, g['lr'] * 1.5)
            print("KL too low # {}".format(kl))
        else:
            print("KL is nice # {}".format(kl))

        # if (update_targ_interval % (update_step_count+1)) == 0:
        model.step_policy_model.policy.load_state_dict(model.train_policy_model.policy.state_dict())
        # model.step_policy_model.cv_net.load_state_dict(model.train_policy_model.cv_net.state_dict())
        model.step_policy_model.vf_net.load_state_dict(model.train_policy_model.vf_net.state_dict())
        model.step_policy_model.policy_optim.zero_grad()
        model.train_policy_model.policy_optim.zero_grad()

        plot_path = 'evals/plots'
        plt_expected_cum_reward(plot_path, exp_rews_means, exp_rews_stds)
        update_step_count += 1

        if save_loc:
            torch.save(model, open(save_loc + "/" + "checkpoint_" + env.spec.id + "_model_.pkl", "wb"))

            if best_exp_mean < mean_reward:
                best_exp_mean = mean_reward
                torch.save(model,
                           open(save_loc + "/" + "best_" + env.spec.id + "_model_" + str(update_step_count) + "_epochs_.pkl", "wb"))

    return model
