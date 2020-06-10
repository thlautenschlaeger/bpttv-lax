import torch
import gym
import numpy as np

from filters import ZFilter
from policies import LaxPolicyModel
from util import calc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()


class Model(object):

    def __init__(self, policy_model: LaxPolicyModel, obs_dim, act_dim, cv_epochs, vf_epochs):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cv_epochs = cv_epochs
        self.vf_epochs = vf_epochs
        self.policy_model = policy_model


    def get_cv_grads(self, obs, actions, rewards, vf_in, values, net_mean, net_std):
        advs = rewards - values

        cv = self.policy_model.cv_net(torch.cat([vf_in, actions], dim=1))
        ddiff_loss = -torch.mean(cv)
        dlogp_mean = (actions - net_mean) / (net_std ** 2)
        dlogp_std = -1 / net_std + 1 / torch.pow(net_std, 3) * torch.square(actions - net_mean)

        pi_loss_mean = -((advs - cv) * dlogp_mean).mean(dim=0)
        pi_loss_std = -((advs - cv) * dlogp_std).mean(dim=0)

        pg_grads_mean = torch.autograd.grad(pi_loss_mean, self.policy_model.policy.net.parameters(),
                                            grad_outputs=torch.ones_like(pi_loss_mean),
                                            create_graph=True, retain_graph=True)
        pg_grads_std = torch.autograd.grad(pi_loss_std, self.policy_model.policy.log_std,
                                           grad_outputs=torch.ones_like(pi_loss_std),
                                           create_graph=True, retain_graph=True)

        ddiff_grads_mean = torch.autograd.grad(ddiff_loss, self.policy_model.policy.net.parameters(),
                                               grad_outputs=torch.ones_like(ddiff_loss),
                                               create_graph=True, retain_graph=True)
        ddiff_grads_std = torch.autograd.grad(ddiff_loss, self.policy_model.policy.log_std,
                                              grad_outputs=torch.ones_like(ddiff_loss),
                                              create_graph=True, retain_graph=True)

        pg_grads_mean = [pg - dg for pg, dg in zip(pg_grads_mean, ddiff_grads_mean)]
        pg_grads_std = [pg - dg for pg, dg in zip(pg_grads_std, ddiff_grads_std)]

        pg_grads = pg_grads_mean + pg_grads_std

        cv_grad = torch.cat([torch.reshape(p, [-1]) for p in pg_grads], 0)
        cv_loss = torch.square(cv_grad).mean()

        cv_grad = torch.autograd.grad(cv_loss, self.policy_model.cv_net.parameters(),
                                      grad_outputs=torch.ones_like(cv_loss))

        return cv_grad

    def update_cv(self, cv_grad):
        self.policy_model.cv_optim.zero_grad()
        for params, cg in zip(self.policy_model.cv_net.parameters(), cv_grad):
            params.backward(cg, create_graph=True, retain_graph=True)
        self.policy_model.cv_optim.step()

    def get_policy_grads(self, obs, actions, rewards, vf_in, value, net_mean, net_std):
        advs = rewards - value

        cv = self.policy_model.cv_net(torch.cat([vf_in, actions], dim=1))
        ddiff_loss = -torch.mean(cv)
        dlogp_mean = (actions - net_mean) / (net_std ** 2)
        dlogp_std = -1 / net_std + 1 / torch.pow(net_std, 3) * torch.square(actions - net_mean)

        pi_loss_mean = -((advs - cv) * dlogp_mean).mean(dim=0)
        pi_loss_std = -((advs - cv) * dlogp_std).mean(dim=0)

        pg_grads_mean = torch.autograd.grad(pi_loss_mean, self.policy_model.policy.net.parameters(),
                                            grad_outputs=torch.ones_like(pi_loss_mean),
                                            create_graph=True, retain_graph=True)
        pg_grads_std = torch.autograd.grad(pi_loss_std, self.policy_model.policy.log_std,
                                           grad_outputs=torch.ones_like(pi_loss_std),
                                           create_graph=True, retain_graph=True)

        ddiff_grads_mean = torch.autograd.grad(ddiff_loss, self.policy_model.policy.net.parameters(),
                                               grad_outputs=torch.ones_like(ddiff_loss),
                                               create_graph=True, retain_graph=True)
        ddiff_grads_std = torch.autograd.grad(ddiff_loss, self.policy_model.policy.log_std,
                                              grad_outputs=torch.ones_like(ddiff_loss),
                                              create_graph=True, retain_graph=True)

        pg_grads_mean = [pg - dg for pg, dg in zip(pg_grads_mean, ddiff_grads_mean)]
        pg_grads_std = [pg - dg for pg, dg in zip(pg_grads_std, ddiff_grads_std)]

        pg_grads = pg_grads_std + pg_grads_mean

        return pg_grads

    def update_vf(self, vf_in, rewards):
        t_loss = 0
        for _ in range(self.vf_epochs):
            pred_reward = self.policy_model.vf_net(vf_in)
            loss = torch.square(pred_reward - rewards).mean()
            self.policy_model.vf_optim.zero_grad()
            loss.backward()
            self.policy_model.vf_optim.step()
            t_loss += loss.detach().cpu().numpy()
        return t_loss / self.vf_epochs

    def update_policy(self, pg):
        self.policy_model.cv_optim.zero_grad()
        for params, _pg in zip(self.policy_model.policy.parameters(), pg):
            params.backward(_pg, create_graph=True, retain_graph=True)
        self.policy_model.policy_optim.step()


class RolloutRunner(object):

    def __init__(self, env: gym.Env, policy_model:LaxPolicyModel, max_path_len, gamma=0.99, lamb=0.97):
        self.env = env
        self.policy_model = policy_model
        self.max_path_len = max_path_len
        self.gamma = gamma
        self.lamb = lamb
        self._num_rollouts = 0
        self._num_steps = 0
        self.obfilter = ZFilter(env.observation_space.shape)

    def run(self):
        ob = self.env.reset()
        ob = self.obfilter(ob)
        ob = to_torch(ob)
        prev_ob = torch.zeros_like(ob)

        done = False

        obs, acts, ac_dist_mean, ac_dist_std, logps, rewards = [], [], [], [], [], []

        for _ in range(self.max_path_len):
            state = torch.cat([ob, prev_ob], -1)
            obs.append(state)
            sampled_ac, log_prob, mean, std = self.policy_model.act(state)
            acts.append(sampled_ac)
            ac_dist_mean.append(mean)
            ac_dist_std.append(std)
            logps.append(log_prob)
            prev_ob = torch.clone(ob)
            scaled_act = self.env.action_space.low + (to_npy(sampled_ac) + 1.) * 0.5 * (self.env.action_space.high - self.env.action_space.low)
            scaled_act = np.clip(scaled_act, a_min=self.env.action_space.low, a_max=self.env.action_space.high)

            ob, reward, done, _ = self.env.step(scaled_act)
            env.render()

            ob = self.obfilter(ob.squeeze())
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
        value = self.policy_model.predict_value(path)
        vtarget = to_torch((calc.discount(np.append(to_npy(rew_t), 0.0 if path["terminated"] else value[-1]), self.gamma)[:-1]).copy())
        vpred_t = torch.cat([value, torch.zeros([1])]) if path["terminated"] else torch.cat([value, value[-1]])
        delta_t = rew_t + self.gamma * vpred_t[1:] - vpred_t[:-1]
        adv_GAE = to_torch(calc.discount(to_npy(delta_t), self.gamma * self.lamb).copy())

        return path, vtarget, value, adv_GAE

def learn(env: gym.Env, policy_model: LaxPolicyModel, seed, total_steps=int(10e6),
          cv_opt_epochs=25, vf_opt_epochs=25, gamma=0.99, lamb=0.97, tsteps_per_batch=2500,
          kl_thresh=0.002):

    max_pathlength = env.spec.max_episode_steps
    obs_dim = policy_model.obs_dim
    act_dim = policy_model.act_dim

    model = Model(policy_model, obs_dim, act_dim, cv_opt_epochs, vf_opt_epochs)
    runner = RolloutRunner(env, policy_model, max_pathlength, gamma=gamma, lamb=lamb)

    tsteps_so_far = 0

    while True:
        if tsteps_so_far > total_steps:
            break

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
            std_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            vf_in = model.policy_model.preproc(path)

            cv_grad = model.get_cv_grads(path["observation"], path["action"], vtarget, vf_in, value,
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
        adv_n = torch.cat(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        rewards_n = torch.cat(vtargs)
        values_n = torch.cat(values)
        values_n = values_n.detach()

        x = torch.cat([model.policy_model.preproc(p) for p in paths])
        x = x.detach()

        vf_loss = model.update_vf(x, rewards_n)
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





if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = LaxPolicyModel(obs_dim, act_dim)

    learn(env, policy, 1337)
