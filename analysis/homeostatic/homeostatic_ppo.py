"""UE 5.5 PPO adapter: complete GAE, centered advantages and task telemetry.

Configure the manager's custom module directory to this folder, trainer filename
homeostatic_ppo. The engine modules remain unchanged.
"""
import time
import numpy as np
import torch
import train_ppo as epic
from ppo import PPOTrainer
from train_common import UE_RESPONSE_SUCCESS


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Bootstrap time limits, stop recursion at either episode boundary."""
    delta = rewards + (~terminated) * gamma * next_values - values
    continuation = (~(terminated | truncated)) * gamma * lam
    advantages = np.empty_like(delta)
    following = 0.0
    for index in range(len(delta) - 1, -1, -1):
        following = delta[index] + continuation[index] * following
        advantages[index] = following
    return advantages, advantages + values


class HomeostaticPPOTrainer(PPOTrainer):
    def compute_advantages_and_returns(self, obs, obs_next, mem, mem_next,
                                      rew, terminated, truncated,
                                      warn_about_advantages=True):
        with torch.no_grad():
            values = self.critic(torch.cat([self.encoder(obs), mem], dim=-1))[..., 0].cpu().numpy()
            next_values = self.critic(torch.cat([self.encoder(obs_next), mem_next], dim=-1))[..., 0].cpu().numpy()
        advantages, returns = gae(rew, values, next_values, terminated, truncated,
                                 self.discount_factor, self.gae_lambda)
        if self.adv_norm:
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)
        advantages = np.clip(advantages, self.adv_min, self.adv_max)
        return (torch.as_tensor(advantages, device=self.policy.device),
                torch.as_tensor(returns, device=self.policy.device), returns.mean())


def setpoint_column(config):
    schema = config['Schemas']['Observations'][0]['Schema']
    elements = schema.get('Elements', {})
    field = elements.get('SetpointDistance')
    if schema.get('Type') != 'And' or not field or field['VectorSize'] != 1:
        raise ValueError('Expected scalar SetpointDistance in the observation schema')
    return sum(x['VectorSize'] for x in elements.values() if x['Index'] < field['Index'])


class TelemetryCommunicator:
    def __init__(self, communicator, config, writer_getter):
        self.communicator = communicator
        self.writer_getter = writer_getter
        self.column = setpoint_column(config)
        self.steps = 0
        self.started = time.monotonic()

    def __getattr__(self, name):
        return getattr(self.communicator, name)

    def receive_experience(self, *args, **kwargs):
        start = time.monotonic()
        result = self.communicator.receive_experience(*args, **kwargs)
        response, buffer, stats = result
        if response == UE_RESPONSE_SUCCESS:
            observations = buffer['obs'][0]
            count = len(observations)
            self.steps += count
            writer = self.writer_getter()
            if writer is not None and count:
                # SpecifyFloatObservation uses FloatScale=100 in BP_AgentInteractor.
                homeostasis = observations[:, self.column] * 100.0
                for tag, value in {
                    'homeostasis/mean_absolute_error': np.abs(homeostasis).mean(),
                    'homeostasis/root_mean_square_error': np.sqrt(np.square(homeostasis).mean()),
                    'homeostasis/fraction_within_5': (np.abs(homeostasis) <= 5.0).mean(),
                    'homeostasis/mean_signed_value': homeostasis.mean(),
                    'throughput/collected_steps': self.steps,
                    'throughput/steps_per_second': self.steps / max(time.monotonic()-self.started, 1e-6),
                    'throughput/collection_seconds': time.monotonic()-start,
                }.items():
                    writer.add_scalar(tag, float(value), self.steps)
                writer.flush()
        return result


def train(config, communicator):
    from torch.utils import tensorboard
    original_writer = tensorboard.SummaryWriter
    original_trainer = epic.PPOTrainer
    writers = []

    class TaskWriter(original_writer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            writers.append(self)

    tensorboard.SummaryWriter = TaskWriter
    epic.PPOTrainer = HomeostaticPPOTrainer
    try:
        proxy = TelemetryCommunicator(communicator, config, lambda: writers[-1] if writers else None)
        epic.train(config, proxy)
    finally:
        for writer in writers:
            writer.close()
        tensorboard.SummaryWriter = original_writer
        epic.PPOTrainer = original_trainer
