import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, n_hidden: int=64, nonlin: str= 'relu', **kwargs):
        super(MLP, self).__init__()
        nlist = dict(relu=nn.ReLU(), tanh=nn.Tanh(),
                     sigmoid=nn.Sigmoid(), softplus=nn.Softplus(), lrelu=nn.LeakyReLU(),
                     elu=nn.ELU())

        self.layer = nn.Linear(in_dim, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, out_dim)
        self.nonlin = nlist[nonlin]

    def forward(self, x, **kwargs):
        x = self.layer(x)
        x = self.nonlin(x)
        x = self.layer2(x)
        x = self.nonlin(x)
        x = self.out(x)

        return x


class GaussianPolicy(nn.Module):

    def __init__(self, in_dim: int, n_hidden: int, out_dim: int,
                 nonlin: str = 'tanh', std=1.0):
        super(GaussianPolicy, self).__init__()

        self.net = MLP(in_dim, out_dim, n_hidden, nonlin)
        # torch.nn.init.uniform_(self.net.out.weight, 0., 0.1)
        # torch.nn.init.uniform_(self.net.out.bias, 0., 0.1)
        self.log_std = nn.Parameter(torch.ones(1, out_dim) * torch.tensor(std).log())

    def forward(self, x, **kwargs):
        x = self.net(x)
        mean = x
        std = self.log_std.exp()

        return mean, std


class LaxPolicyModel:

    def __init__(self, obs_dim, act_dim, p_hidden=64, vf_hidden=64, cv_hidden=64, p_lr=1e-4, vf_lr=1e-3, cv_lr=1e-3):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.p_hidden=p_hidden
        self.vf_hidden = vf_hidden
        self.cv_hidden = cv_hidden
        self.p_lr = p_lr
        self.vf_lr = vf_lr
        self.cv_lr = cv_lr
        self.vf_in_dim = self.obs_dim * 2 + self.act_dim * 2 + 2
        self.cv_in_dim = self.vf_in_dim + self.act_dim
        self.policy_in_dim = self.obs_dim * 2

        self.cv_net = MLP(in_dim=self.cv_in_dim, out_dim=1, n_hidden=cv_hidden, nonlin='elu')
        self.vf_net = MLP(in_dim=self.vf_in_dim, out_dim=1, n_hidden=vf_hidden, nonlin='elu')
        self.policy = GaussianPolicy(in_dim=self.policy_in_dim, n_hidden=p_hidden, out_dim=self.act_dim, nonlin='tanh')

        self.cv_optim = torch.optim.Adam(self.cv_net.parameters(), lr=self.cv_lr)
        self.vf_optim = torch.optim.Adam(self.vf_net.parameters(), lr=self.vf_lr)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.p_lr)

    def act(self, obs):
        mean, std = self.policy(obs)
        std = std.squeeze(0)
        dist = torch.distributions.Normal(mean, std)
        sampled_ac = dist.rsample()
        log_prob = dist.log_prob(sampled_ac)

        return sampled_ac, log_prob, mean, std

    def value(self, x):
        return self.vf_net(x).squeeze(1)

    def predict_value(self, path):
        return self.value(self.preproc(path))

    @staticmethod
    def preproc(path):
        path_len = len(path['reward'])
        al = torch.arange(path_len).reshape(-1, 1)/10.0
        act_mean = path["act_mean"]
        act_std = path["act_std"]
        x = torch.cat([path['observation'], act_mean, act_std, al, torch.ones((path_len, 1))], dim=1)
        return x.detach()

    def compute_kl(self, obs, old_act_mu, old_act_std):
        params = self.policy(obs)
        old_dist = torch.distributions.Normal(old_act_mu, old_act_std)
        new_dist = torch.distributions.Normal(*params)
        return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
