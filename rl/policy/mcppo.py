import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Optional

from rl.data import Batch, RunningMeanStd
from .ppo1 import split, gae_return, stack_index


class MCPPOPolicy:
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
            multi_critic: bool = True,
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
        # self._w_adv = torch.from_numpy(np.array([0.4, 0.6])[:, None]).float().to(device)
        self._target_kl = target_kl
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1.0, "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._rew_norm = reward_normalization
        self._multi_critic = multi_critic
        if not self._rew_norm:
            assert not self._value_clip, "value clip is available only when `reward_normalization` is True"
        self._adv_norm = advantage_normalization
        self._recompute_adv = recompute_advantage
        self.device = device
        self._eps = 1e-8
        self.task_dim = getattr(critic, 'task_dim', 1)
        assert self.task_dim > 1
        self.ret_rms = [RunningMeanStd() for _ in range(self.task_dim)]
        self.actor_stack_dim = getattr(actor, 'stack_dim', 1)
        self.critic_stack_dim = getattr(critic, 'stack_dim', 1)
        if self.actor_stack_dim > 1 and self.critic_stack_dim > 1:
            assert self.actor_stack_dim == self.critic_stack_dim
        self.task_data_size = None

    def process_fn(self, batch: Batch) -> Batch:
        batch.obs = torch.from_numpy(batch.obs).float().to(self.device)
        batch.obs_next = torch.from_numpy(batch.obs_next).float().to(self.device)
        batch.act = torch.from_numpy(batch.act).float().to(self.device)
        # batch.rew = batch.rew.transpose()
        if self.actor_stack_dim > 1 or self.critic_stack_dim > 1:
            indices = stack_index(np.concatenate([[0], np.where(batch.done)[0] + 1, [len(batch.done)]]),
                                  self.actor_stack_dim)
            batch.indices = torch.from_numpy(indices).long().to(self.device)
        batch = self._compute_returns(batch)
        with torch.no_grad():
            obs = batch.obs[batch.indices] if self.actor_stack_dim > 1 else batch.obs
            batch.logp_old = self.actor(obs)['dist'].log_prob(batch.act)
        self.task_data_size = np.asarray([sum(batch.info.task_id == i) for i in range(self.task_dim)])
        return batch

    def _compute_returns(self, batch: Batch) -> Batch:
        obs = batch.obs[batch.indices] if self.critic_stack_dim > 1 else batch.obs
        obs_next = batch.obs_next[batch.indices] if self.critic_stack_dim > 1 else batch.obs_next
        batch.v_s = torch.zeros(len(batch.obs), dtype=torch.float32, device=self.device)
        batch.adv = torch.zeros_like(batch.v_s)
        batch.ret = torch.zeros_like(batch.v_s)
        ret_rms_var, ret_rms_mean = 0, 0
        if self._multi_critic:
            for i in range(self.task_dim):
                ret_rms_var += self.ret_rms[i].var / 7.
                ret_rms_mean += self.ret_rms[i].mean / 7.
        else:
            ret_rms_var, ret_rms_mean = self.ret_rms[0].var, self.ret_rms[0].mean
        for i in range(self.task_dim):
            mask = (batch.info.task_id == i)
            if any(mask):
                with torch.no_grad():
                    v_s = self.critic(obs[mask], net_id=i).flatten()
                    v_s_ = self.critic(obs_next[mask], net_id=i).flatten()
                    batch.v_s[mask], v_s, v_s_ = v_s, v_s.cpu().numpy(), v_s_.cpu().numpy()
                if self._rew_norm:  # unnormalize v_s & v_s_
                    var = np.sqrt(ret_rms_var + self._eps)
                    v_s = v_s * var
                    v_s_ = v_s_ * var
                adv = gae_return(v_s, v_s_ * batch.info.value_mask[mask], batch.rew[mask], batch.info.reward_mask[mask], self._gamma, self._lambda)
                _ret = adv + v_s
                if self._rew_norm:  # normalize ret
                    ret = _ret / np.sqrt(ret_rms_var + self._eps)
                    self.ret_rms[i].update(_ret)
                else:
                    ret = _ret
                adv = torch.from_numpy(adv).float().to(self.device)
                if self._adv_norm:
                    mean, std = adv.mean(), adv.std()
                    adv = (adv - mean) / std  # per-batch norm
                batch.adv[mask] = adv * self.critic.task_adv_coef[i]
                batch.ret[mask] = torch.from_numpy(ret).float().to(self.device)
        if self._adv_norm and self._multi_critic:
            mean, std = batch.adv.mean(), batch.adv.std()
            batch.adv = (batch.adv - mean) / std  # per-batch adv norm
        return batch

    def _optimize_actor(self, dist, act, logp_old, adv):
        logp = dist.log_prob(act)
        ratio = (logp - logp_old).exp().float()
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
        return clip_loss.item(), ent_loss.item(), approx_kl

    def _optimize_critic(self, value, ret, v_s, net_id):
        if self._value_clip:
            v_clip = v_s + (value - v_s).clamp(-self._eps_clip, self._eps_clip)
            vf1 = (ret - value).pow(2)
            vf2 = (ret - v_clip).pow(2)
            vf_loss = torch.max(vf1, vf2).mean()
        else:
            vf_loss = (ret - value).pow(2).mean()
        loss_coef = self.critic.task_adv_coef[net_id]
        critic_loss = (self._w_vf * loss_coef * vf_loss)
        # critic_loss = self._w_vf * vf_loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self._max_grad_norm:  # clip large gradient
            nn.utils.clip_grad_norm_(
                list(self.critic.parameters()),
                max_norm=self._max_grad_norm)
        self.critic_optim.step()
        return vf_loss.item()

    def learn(self, batch: Batch, batch_size: int, repeat: int) -> Dict[str, List[float]]:
        vf_losses, clip_losses, ent_losses = [], [], []
        # with torch.autograd.detect_anomaly():
        batch_indices = np.arange(len(batch.obs))
        for step in range(repeat):
            # ----------------optimize critic----------------------------------
            vf_loss = []
            for i in range(self.task_dim):
                mask = (batch.info.task_id == i)
                if any(mask):
                    for ind in split(batch_indices[mask], batch_size, shuffle=True, merge_last=True):
                        ind = torch.from_numpy(ind).long().to(self.device)
                        critic_obs = batch.obs[batch.indices[ind]] if self.critic_stack_dim > 1 else batch.obs[ind]
                        ret, v_s = batch.ret[ind], batch.v_s[ind]
                        value = self.critic(critic_obs, net_id=i).flatten()
                        # vf_losses.append(self._optimize_critic(value, ret, v_s, i))
                        vf_loss.append(self._optimize_critic(value, ret, v_s, i))
            vf_losses.append(vf_loss)
            # --------------optimize actor-----------------------------------
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch)
            if self._multi_critic:
                actor_batch_size = 4 * batch_size
            else:
                actor_batch_size = batch_size
            for ind in split(batch_indices, actor_batch_size, shuffle=True, merge_last=True):
                ind = torch.from_numpy(ind).long().to(self.device)
                actor_obs = batch.obs[batch.indices[ind]] if self.actor_stack_dim > 1 else batch.obs[ind]
                act, logp_old, adv = batch.act[ind], batch.logp_old[ind], batch.adv[ind]
                if self._adv_norm:
                    mean, std = adv.mean(), adv.std()
                    adv = (adv - mean) / std  # per-batch norm
                dist = self.actor(actor_obs)['dist']
                clip_loss, ent_loss, approx_kl = self._optimize_actor(dist, act, logp_old, adv)
                clip_losses.append(clip_loss)
                ent_losses.append(ent_loss)
            # early stopping
            if approx_kl > self._target_kl:
                break
        return {
            "vf": vf_losses,
            "clip": clip_losses,
            "ent": ent_losses,
            'kl': approx_kl,
            'repeat': step + 1,
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
