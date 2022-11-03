import torch
from numba import njit
import numpy as np
from torch import nn
from typing import Any, Dict, List, Optional

from rl.data import Batch, RunningMeanStd


class PPOPolicy:
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network.
    :param torch.nn.Module critic: the critic network.
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1. Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation. Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            critic: torch.nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic_optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            max_grad_norm: Optional[float] = None,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            eps_clip: float = 0.2,
            target_kl: float = 0.015,
            gae_lambda: float = 0.95,
            dual_clip: Optional[float] = None,
            value_clip: bool = True,
            reward_normalization: bool = True,
            advantage_normalization: bool = True,
            recompute_advantage: bool = False,
            device: str = 'cpu'
    ) -> None:
        self._compile()
        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        assert (0.0 <= discount_factor <= 1.0), "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._target_kl = target_kl
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1.0, "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._rew_norm = reward_normalization
        if not self._rew_norm:
            assert not self._value_clip, "value clip is available only when `reward_normalization` is True"
        self._adv_norm = advantage_normalization
        self._recompute_adv = recompute_advantage
        self.device = device
        self._eps = 1e-8
        self.ret_rms = RunningMeanStd()
        self.actor_stack_dim = getattr(actor, 'stack_dim', 1)
        self.critic_stack_dim = getattr(critic, 'stack_dim', 1)
        if self.actor_stack_dim > 1 and self.critic_stack_dim > 1:
            assert self.actor_stack_dim == self.critic_stack_dim

    def process_fn(self, batch: Batch) -> Batch:
        batch.obs = torch.from_numpy(batch.obs).float().to(self.device)
        batch.obs_next = torch.from_numpy(batch.obs_next).float().to(self.device)
        batch.act = torch.from_numpy(batch.act).float().to(self.device)
        if self.actor_stack_dim > 1 or self.critic_stack_dim > 1:
            indices = stack_index(np.concatenate([[0], np.where(batch.done)[0] + 1, [len(batch.done)]]), self.actor_stack_dim)
            batch.indices = torch.from_numpy(indices).long().to(self.device)
        batch = self._compute_returns(batch)
        with torch.no_grad():
            obs = batch.obs[batch.indices] if self.actor_stack_dim > 1 else batch.obs
            batch.logp_old = self.actor(obs)['dist'].log_prob(batch.act)
        return batch

    def _compute_returns(self, batch: Batch) -> Batch:
        with torch.no_grad():
            obs = batch.obs[batch.indices] if self.critic_stack_dim > 1 else batch.obs
            obs_next = batch.obs_next[batch.indices] if self.critic_stack_dim > 1 else batch.obs_next
            v_s = self.critic(obs).flatten()
            v_s_ = self.critic(obs_next).flatten()
            batch.v_s, v_s, v_s_ = v_s, v_s.cpu().numpy(), v_s_.cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            var = self.ret_rms.var
            var = np.sqrt(var + self._eps)
            v_s = v_s * var
            v_s_ = v_s_ * var
        adv = gae_return(v_s, v_s_ * batch.info.value_mask, batch.rew, batch.info.reward_mask, self._gamma, self._lambda)
        _ret = adv + v_s
        if self._rew_norm:  # normalize ret
            ret = _ret / var
            self.ret_rms.update(_ret)
        else:
            ret = _ret
        batch.adv = torch.from_numpy(adv).float().to(self.device)
        batch.ret = torch.from_numpy(ret).float().to(self.device)
        return batch

    def learn(self, batch: Batch, batch_size: int, repeat: int) -> Dict[str, List[float]]:
        vf_losses, clip_losses, ent_losses = [], [], []
        # actor_repeat =critic_repeat= repeat
        # actor_early_stopping = False
        # with torch.autograd.detect_anomaly():
        batch_indices = np.arange(len(batch))
        for step in range(repeat):
            # optimize critic
            for ind in split(batch_indices, batch_size, shuffle=True, merge_last=True):
                ind = torch.from_numpy(ind).long().to(self.device)
                critic_obs = batch.obs[batch.indices[ind]] if self.critic_stack_dim > 1 else batch.obs[ind]
                ret, v_s = batch.ret[ind], batch.v_s[ind]
                value = self.critic(critic_obs).flatten()
                if self._value_clip:
                    v_clip = v_s + (value - v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (ret - value).pow(2)
                    vf2 = (ret - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (ret - value).pow(2).mean()
                critic_loss = self._w_vf * vf_loss
                self.critic_optim.zero_grad()
                critic_loss.backward()
                if self._max_grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        list(self.critic.parameters()),
                        max_norm=self._max_grad_norm)
                self.critic_optim.step()
                vf_losses.append(vf_loss.item())
                # # early stopping
                # if approx_kl > self._target_kl:
                #     break
                #     # actor_early_stopping = True
                #     # actor_repeat = step + 1
            # optimize actor
            # if not actor_early_stopping:
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch)
            for ind in split(batch_indices, batch_size, shuffle=True, merge_last=True):
                ind = torch.from_numpy(ind).long().to(self.device)
                actor_obs = batch.obs[batch.indices[ind]] if self.actor_stack_dim > 1 else batch.obs[ind]
                act, logp_old, adv = batch.act[ind], batch.logp_old[ind], batch.adv[ind]
                if self._adv_norm:
                    mean, std = adv.mean(), adv.std()
                    adv = (adv - mean) / std  # per-batch norm
                dist = self.actor(actor_obs)['dist']
                logp = dist.log_prob(act)
                ratio = (logp - logp_old).exp().float()
                # ratio = (dist.log_prob(act) - logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * adv
                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2), self._dual_clip * adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                approx_kl = torch.mean(logp_old - logp).item()
                ent_loss = dist.entropy().mean()
                actor_loss = clip_loss - self._w_ent * ent_loss
                self.actor_optim.zero_grad()
                actor_loss.backward()
                if self._max_grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()),
                        max_norm=self._max_grad_norm)
                self.actor_optim.step()
                clip_losses.append(clip_loss.item())
                ent_losses.append(ent_loss.item())
            # early stopping
            if approx_kl > self._target_kl:
                break
                # actor_early_stopping = True
                # actor_repeat = step + 1
        return {
            "vf": vf_losses,
            "clip": clip_losses,
            "ent": ent_losses,
            'kl': approx_kl,
            'repeat': step + 1,
            # 'critic_repeat': critic_repeat,
        }

    def update(self, batch: Batch, batch_size: int, repeat: int) -> Dict[str, Any]:
        batch = self.process_fn(batch)
        return self.learn(batch, batch_size, repeat)

    def update_actor_learning_rate(self, v):
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = v

    def update_critic_learning_rate(self, v):
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = v

    def _compile(self) -> None:
        i64 = np.array([0, 1], dtype=np.int64)
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([True, False], dtype=np.bool_)
        gae_return(f64, f64, f64, b, 0.1, 0.1)
        gae_return(f32, f32, f64, b, 0.1, 0.1)
        stack_index(i64, 1)


def split(indices: np.ndarray, batch_size: int, shuffle: bool = True, merge_last: bool = False) -> np.ndarray:
    assert 1 <= batch_size  # size can be greater than length, return whole batch
    length = len(indices)
    if shuffle:
        indices = np.random.permutation(indices)
    merge_last = merge_last and length % batch_size > 0
    for idx in range(0, length, batch_size):
        if merge_last and idx + batch_size + batch_size >= length:
            yield indices[idx:]
            break
        yield indices[idx:idx + batch_size]


@njit
def gae_return(
        v_s: np.ndarray,
        v_s_: np.ndarray,
        rew: np.ndarray,
        mask: np.ndarray,
        gamma: float,
        gae_lambda: float,
) -> np.ndarray:
    adv = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    m = mask * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + m[i] * gae
        adv[i] = gae
    return adv


@njit
def stack_index(episode_start_index: np.ndarray, stack_dim: int):
    num = 0
    indices = np.empty((episode_start_index[-1], stack_dim), dtype=np.int64)
    for i in range(len(episode_start_index) - 1):
        start, end = episode_start_index[i], episode_start_index[i + 1]
        for j in range(start, end):
            j += 1
            size = start - (j - stack_dim)
            indices[num] = np.concatenate((np.full(size, start, dtype=np.int64), np.arange(start, j))) if size > 0 else np.arange(j - stack_dim, j)
            num += 1
    return indices


if __name__ == '__main__':
    import time

    done_index = np.array([2, 7, 15], dtype=np.int64)
    episode_start_index = np.concatenate([[0], done_index + 1, [20]])
    indices = stack_index(episode_start_index, 5)

    done_index = np.sort(np.random.randint(0, 40000, 150))
    episode_start_index = np.concatenate([[0], done_index + 1, [40000]])
    t0 = time.time()
    for _ in range(10):
        indices = stack_index(episode_start_index, 100)
    print((time.time() - t0) / 10.)
