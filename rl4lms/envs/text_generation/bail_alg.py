from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import random
import os
import torch
import torch.nn.functional as F

from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecEnv
from transformers import PreTrainedTokenizer

from rl4lms.algorithms.common.maskable.buffers import MaskableDictRolloutBuffer
from rl4lms.envs.text_generation.kl_controllers import KLController
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.policy.base_policy import (
    PolicyOutput,
    RefPolicyOutput,
    ValueOutput,
)
from rl4lms.envs.text_generation.reward import BatchedRewardFunction, RewardFunction
from rl4lms.envs.text_generation.warm_start import OnPolicyWarmStartMixin

from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.alg_wrappers import unpack_observations


@dataclass
class BAILTransitionInfo:
    observation: TensorDict
    action: np.ndarray
    reward: np.ndarray
    episode_start: np.ndarray
    value: torch.Tensor
    log_prob: torch.Tensor
    done: np.ndarray
    action_mask: np.ndarray
    info: Dict[str, Any]


def compute_batched_rewards(
    episode_wise_transitions: List[BAILTransitionInfo], reward_fn: RewardFunction
):
    # first collect all the prompts, ref and gen texts
    prompts = []
    reference_texts = []
    generated_texts = []
    is_dones = []
    indices = []
    meta_infos = []
    for trans_ix, transition in enumerate(episode_wise_transitions):
        done = transition.done
        info = transition.info
        prompts.append(info["prompt_text"])
        reference_texts.append(info["reference_text"])
        generated_texts.append(info["output"])
        is_dones.append(done)
        meta_infos.append(info["meta_info"])
        indices.append(trans_ix)

    # compute rewards all at once
    rewards = reward_fn(prompts, generated_texts, reference_texts, is_dones, meta_infos)
    # rewards = rewards.numpy().flatten()

    # override the rewards in transitions
    for trans_ix, reward in zip(indices, rewards):
        episode_wise_transitions[trans_ix].reward = np.array([reward])

class BAIL:
    def __init__(self,
                 tracker,
                 env,
                 policy_cls,
                 policy_kwargs,
                 device,
                 gamma=0.99,
                 buffer_size=int(1e6),
                 seed=0,
                 norm_reward=False
                 ):

        random.seed(seed)

        self.norm_reward = norm_reward
        self.env = env
        self.tracker = tracker
        self.tokenizer = self.env.tokenizer

        self.device = device
        self.gamma = gamma
        self.buffer_size = buffer_size

        self.selected_indices = None
        self.selected_len = 0
        self.border = None

        self.policy = policy_cls(
            env.observation_space,
            env.action_space,
            lr_schedule=None,
            **policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        self.rollout_buffer = MaskableDictRolloutBuffer(
            self.buffer_size,
            env.observation_space,
            env.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=1,
            n_envs=1,
        )

        self.reward_fn = self.env.reward_function

    # -------------------------------------------------------------------------------------
    def get_state_with_mc_return(self):
        self._fill_buffer()
        if not self.rollout_buffer.full:
            self._calculate_mc_return(done=True)

    def _fill_buffer(self, rollout_info: Dict[str, Any] = None):
        while self.env.sampler_for_replaying.size() > 0:
            # if rollout buffer is already full, do not continue
            if self.rollout_buffer.full:
                return

            sample = self.env.sampler_for_replaying.pop()
            gen_texts = sample.references[0]
            current_obs = self.env.reset(sample)
            episode_starts = np.ones((1, ), dtype=bool)

            # prepare generations
            gen_tokens = self.tokenizer.encode(
                gen_texts,
                truncation=True,
                max_length=self.env.max_steps,
                return_tensors='pt')
            step_wise_actions = [gen_tokens[:, step] for step in range(gen_tokens.shape[-1])]  # Need Tensor here.

            # process them one step at a time to collect rollout info
            episode_wise_transitions = []
            ep_terminated = False

            for action_tensor in step_wise_actions:
                # if all episodes are done, just break and do not continue
                if ep_terminated:
                    break
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(current_obs, self.device)
                # step into env to get rewards
                action = action_tensor.cpu().numpy()
                new_obs, reward, done, info = self.env.step(action.item())

                unpacked_obs = unpack_observations(obs_tensor, 1)

                # store episode wise transitions separately
                # only if not terminated already
                transition = BAILTransitionInfo(
                    observation=unpacked_obs[0],
                    action=action,
                    reward=np.array([reward]),
                    episode_start=episode_starts,
                    value=torch.zeros(1),
                    log_prob=torch.zeros(1),
                    done=np.array([done], dtype=int),
                    action_mask=None,
                    info=info,
                )

                episode_wise_transitions.append(transition)

                # mark this episode to terminated if done occurs once
                if done:
                    ep_terminated = True

                episode_starts = np.zeros((1,), dtype=bool)
                current_obs = new_obs

            # !!! Severe Bug !!!
            if episode_wise_transitions[-1].done.item() == 0:
                continue
            if isinstance(self.reward_fn, BatchedRewardFunction):
                compute_batched_rewards(episode_wise_transitions, self.reward_fn)

            return_computed = False
            for transition_ix, transition in enumerate(episode_wise_transitions):
                if not self.rollout_buffer.full:
                    self.rollout_buffer.add(
                        transition.observation,
                        transition.action,
                        transition.reward,
                        transition.episode_start,
                        transition.value,
                        transition.log_prob,
                        action_masks=transition.action_mask,
                    )

                if self.rollout_buffer.full and not return_computed:
                    self._calculate_mc_return(transition.done)
                    return_computed = True

    def _calculate_mc_return(self, done):
        # normalize the rewards
        if self.norm_reward:
            mean = self.rollout_buffer.rewards.mean()
            std = self.rollout_buffer.rewards.std()
            self.rollout_buffer.rewards = (self.rollout_buffer.rewards - mean) / (
                    std + 1e-8
            )

        g = 0
        length = self.rollout_buffer.size()
        for step in range(length - 1, -1, -1):
            g = self.rollout_buffer.rewards[step] if done else g * self.gamma + self.rollout_buffer.rewards[step]
            self.rollout_buffer.returns[step] = g
            done = self.rollout_buffer.episode_starts[step]

    # -------------------------------------------------------------------------------------
    def learn(self, seed=0, ue_lr=3e-3, ue_wd=2e-2, ue_loss_k=1000, ue_train_epoch=50, ue_batch_size=800,
              ue_consecutive_steps=4, ue_clip=None, pct=0.25):

        print("---------------------------------------")
        print("Start learning with BAIL")
        print("---------------------------------------")

        self.get_state_with_mc_return()
        print('Finish loading states with mc returns type with Gamma:', self.gamma,
              '; Max Episode Length:', self.env.max_steps,
              '; Total rollouts:', self.rollout_buffer.size(), '/', self.buffer_size)

        if not os.path.exists(
                f'./pytorch_models/Stat_UE_Vmodel_K{ue_loss_k}.pth')\
            and not os.path.exists(
                f'./pytorch_models/Stat_UE_Vhead_K{ue_loss_k}.pth'):
            # train upper envelope
            print('')
            print('---------- UE train starts ---------')
            print('With testing MClength:', self.env.max_steps, 'training loss ratio k:', ue_loss_k)
            self.train_upper_envelope(seed=seed,
                                      learning_rate=ue_lr,
                                      weight_decay=ue_wd,
                                      num_epoches=ue_train_epoch,
                                      batch_size=ue_batch_size,
                                      consecutive_steps=ue_consecutive_steps,
                                      k=ue_loss_k)
            torch.save(self.policy._value_model.state_dict(), f'./pytorch_models/Stat_UE_Vmodel_K{ue_loss_k}.pth')
            torch.save(self.policy._value_head.state_dict(), f'./pytorch_models/Stat_UE_Vhead_K{ue_loss_k}.pth')
        else:
            print('---------- Load pretrained Upper Envelope model -----------')
            self.policy._value_model.load_state_dict(torch.load(
                f'./pytorch_models/Stat_UE_Vmodel_K{ue_loss_k}.pth'))
            self.policy._value_head.load_state_dict(torch.load(
                f'./pytorch_models/Stat_UE_Vhead_K{ue_loss_k}.pth'))
            print('Successfully load seed %s envelope from' % seed, 'with training loss ratio k:', ue_loss_k)

        print('----------- Doing selection in Buffer via ue ----------')
        self.selected_indices, self.selected_len, self.border = self.select_batch_ue(C=ue_clip, select_percentage=pct)

    # -------------------------------------------------------------------------------------
    def train_upper_envelope(self, seed,
                             learning_rate=3e-3,
                             weight_decay=0.02,
                             num_epoches=50,
                             batch_size=800,
                             consecutive_steps=4,
                             k=10000):
        print('')
        print('Setting UE training optimizer:')
        params = list(self.policy._value_model.named_parameters()) + list(self.policy._value_head.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate
        )
        print('UE training optimizer set: Done.')

        # Split data into training and testing #
        # But make sure the highest Ri is in the training set
        # pick out the highest data point
        length = self.rollout_buffer.size()
        highest_idx = np.argmax(self.rollout_buffer.returns, axis=0).item()
        print("Highest Return Index:", highest_idx, '; Highest Return:', self.rollout_buffer.returns[highest_idx])

        remaining_idx = list(range(0, highest_idx)) + list(range(highest_idx+1, length))
        random.shuffle(remaining_idx)

        # divide data into train/test
        divide = int(length * 0.8)
        train_idx = remaining_idx[:divide]
        test_idx = remaining_idx[divide:]

        # add the highest data into training
        train_idx.append(highest_idx)

        # train upper envelope
        num_increase = 0
        previous_loss = float('inf')
        best_value_model_parameters = self.policy._value_model.state_dict()
        best_value_head_parameters = self.policy._value_head.state_dict()

        # Upper Envelope Training starts
        self.policy.set_training_mode(True)
        total_batch = len(train_idx) // batch_size
        print_freq = total_batch // 6

        print('UE training starts:')
        for epoch in range(num_epoches):
            train_loss = 0
            random.shuffle(train_idx)

            for batch_ix, rollout_data in \
                    enumerate(list(self.rollout_buffer.get_from_idxs(batch_size, np.array(train_idx)))):
                value_output = self.policy.forward_value(rollout_data.observations)
                values = value_output.values
                values = values.flatten()
                loss = self._L2PenaltyLoss(values, rollout_data.returns.flatten(), k_val=k)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_ix % print_freq == 0:
                    print(f"Epoch: {epoch}/{num_epoches}; Batch: {batch_ix}/{total_batch}; BatchLoss: {loss.item()}")

            # early stopping

            # calculate validation error
            validation_loss = self._calc_ue_valiloss(test_idx, ue_bsize=batch_size, ue_loss_k=k)
            if validation_loss < previous_loss:
                previous_loss = validation_loss
                best_value_model_parameters = self.policy._value_model.state_dict()
                best_value_head_parameters = self.policy._value_head.state_dict()
                num_increase = 0
            else:
                num_increase += 1
            if num_increase == consecutive_steps:
                self.policy._value_model.load_state_dict(best_value_model_parameters)
                self.policy._value_head.load_state_dict(best_value_head_parameters)

            print()
            print('UE Training Epoch:', epoch + 1)
            print('UETrainLoss:', train_loss)
            print('UEValLoss:', validation_loss.cpu().item())

        print("Upper Envelope training is complete.")

    def _L2PenaltyLoss(self, predicted, target, k_val):
        # perm = np.arange(predicted.shape[0])
        # loss = torch.zeros(1, requires_grad=True).to(self.device)
        # num = 0
        # for i in perm:
        #     Vsi = predicted[i]
        #     yi = target[i]
        #     if Vsi >= yi:
        #         mseloss = (Vsi - yi) ** 2
        #     else:
        #         mseloss = k_val * (yi - Vsi) ** 2
        #         num += 1
        #     loss = torch.add(loss, mseloss)  # a very big number
        mask = k_val * (predicted < target)
        loss = (mask * (predicted - target) ** 2).mean()
        return loss

    def _calc_ue_valiloss(self, test_idxs, ue_bsize, ue_loss_k):
        print('Evaluating UE on validation set:')
        self.policy.set_training_mode(False)
        with torch.no_grad():
            validation_loss = torch.FloatTensor([0]).detach().to(self.device)
            for batch_idx, rollout_data \
                    in enumerate(list(self.rollout_buffer.get_from_idxs(ue_bsize, np.array(test_idxs)))):
                value_output = self.policy.forward_value(rollout_data.observations)
                values = value_output.values
                values = values.flatten()
                loss = self._L2PenaltyLoss(values, rollout_data.returns.flatten(), k_val=ue_loss_k)
                validation_loss += loss.item()
        self.policy.set_training_mode(True)
        return validation_loss

    # -------------------------------------------------------------------------------------
    def select_batch_ue(self, C, select_percentage):
        print('Calculating UE for each data in the buffer:')
        self.policy.set_training_mode(False)
        num_data = self.rollout_buffer.size()
        ratios = torch.zeros(1).to(self.device)
        with torch.no_grad():
            batch_size = 32
            indices = np.arange(0, num_data)
            for i, rollout_data in enumerate(list(self.rollout_buffer.get_from_idxs(batch_size, indices))):
                values = self.policy.forward_value(rollout_data.observations).values.flatten()
                # self.rollout_buffer.values[i] = value.squeeze().item()
                if C is None:
                    ratios = torch.cat((ratios, rollout_data.returns / values))
                else:
                    ratios = torch.cat((ratios, rollout_data.returns / torch.minimum(values, torch.tensor(C))))

        # ratios = torch.stack(ratios).view(-1)
        increasing_ratios, increasing_ratio_indices = torch.sort(ratios)
        bor_ind = increasing_ratio_indices[-int(select_percentage * num_data)]
        border = ratios[bor_ind]

        print('Begin selecting with ue border:', border.item())
        selected_idx = torch.nonzero((ratios - border) >= 0).flatten().tolist()
        # for i in range(num_data):
        #     rat = ratios[i]
        #     if rat >= border:
        #         selected_idx.append(i)

        initial_len, selected_len = num_data, len(selected_idx)
        print('Finish selecting with Border:', border, '; Selecting ratio:', selected_len, '/', initial_len)

        return selected_idx, selected_len, border

    # -------------------------------------------------------------------------------------
    def BC(self, optimizer, n_batch, batch_size, epoch, beta):
        self.policy.set_training_mode(True)
        total_loss = 0
        print_freq = len(self.selected_indices) // batch_size // 10
        # for it in range(n_batch):
            # batch_idxs = random.sample(self.selected_indices, batch_size)
            # TBD: Write another _get_samples function in buffer.py instead of overwriting the original one!
            # rollout_data = self.rollout_buffer._get_samples(np.array(batch_idxs))
        for batch_ix, rollout_data in enumerate(list(self.rollout_buffer.get_from_idxs(batch_size, self.selected_indices))):
            loss, kl, nll = self._KLPenaltyLoss(rollout_data, batch_size, beta)
            total_loss += loss.item()

            # Optimize the actor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_ix % print_freq == 0:
                print(f"Epoch: {epoch}; Batch_ix: {batch_ix}; TrainLoss: {loss.item()}; KL: {kl.item()}; NLL: {nll.item()}")
        self.policy.set_training_mode(False)
        return total_loss / (batch_ix + 1)

    # TBD: Make the following work also for Seq2SeqModel. Especially, use self.policy._action_dist attribute to imporve
    # current KL calculation.
    def _KLPenaltyLoss(self, rollout_data, batch_size, beta=0.05):
        obs = rollout_data.observations
        actions = rollout_data.actions
        policy_model = self.policy._policy_model
        ref_model = self.policy._ref_model.eval()

        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]
        past_model_kwargs = {
            "attention_mask": attention_mask,
        }

        model_inputs = self.policy._prepare_inputs_for_model(
            policy_model, input_ids, past_model_kwargs
        )

        policy_output = policy_model(output_hidden_states=True, **model_inputs)
        ref_output = ref_model(output_hidden_states=True, **model_inputs)

        policy_logits = policy_output.logits[:, -1, :]
        ref_logits = ref_output.logits[:, -1, :]


        kl = F.kl_div(F.log_softmax(policy_logits, dim=-1), F.log_softmax(ref_logits, dim=-1),
                      log_target=True, reduction='batchmean')
        dist = self.policy._action_dist.proba_distribution(action_logits=policy_logits)
        policy_log_prob = dist.log_prob(actions)

        loss = -policy_log_prob.mean() + beta * kl

        return  loss, kl, -policy_log_prob.mean()



