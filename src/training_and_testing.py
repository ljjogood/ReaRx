import os
import numpy as np
from sympy import refine
from tqdm import trange
from param_parser import parameter_parser, tab_printer
from model import FHPRModel, TCMHeteroGNN
import torch
import torch.nn as nn
import torch.nn.functional as F


class FHPRTrainer:
    """Stage 1: Foundation herbal prescription reasoning module trainer"""

    def __init__(self, args, herb_numbers,sym_guidelines, dis_guidelines):
        tab_printer(args)
        self.args = args
        self.herb_numbers = herb_numbers
        self.sym_guidelines = sym_guidelines
        self.dis_guidelines = dis_guidelines
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_models()

    def _setup_models(self):
        self.graphEmbModel = TCMHeteroGNN(args=self.args).to(self.device)
        self.model_g = FHPRModel(
            args=self.args,
            num_herbs=self.herb_numbers,
            sym_rule=self.sym_guidelines,
            dis_rule=self.dis_guidelines,
        ).to(self.device)

        self.optimizer_g = torch.optim.Adam(
            list(self.graphEmbModel.parameters()) +
            list(self.model_g.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        self.loss_fcn1 = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_fcn2 = nn.MSELoss(reduction='mean')
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss()

    def train(self, train_loader, test_loader, graph_data, global_dosage_min, global_dosage_range):
        print("=" * 60)
        print("Stage 1: Foundation Herbal Prescription Model Training ")
        print("=" * 60)
        print("device:", self.device)

        self.global_dosage_min = global_dosage_min
        self.global_dosage_range = global_dosage_range

        x = {k: v.to(self.device) for k, v in graph_data.x_dict.items()}
        edge_index = {k: v.to(self.device) for k, v in graph_data.edge_index_dict.items()}

        epochs = trange(self.args.epochs_sl, leave=True, desc="Epoch")
        count = 0
        best_loss = 100

        for epoch in epochs:
            training_losses = 0.0
            self.model_g.train()
            self.graphEmbModel.train()
            train_batches = len(train_loader)

            for sym_ori, syn_ori, dis_ori, herb_sim, dosage_vector, true_herb, true_dosage in train_loader:
                self.optimizer_g.zero_grad()
                embeddings = self.graphEmbModel(x, edge_index)

                sym_ori = sym_ori.to(self.device)
                syn_ori = syn_ori.to(self.device)
                dis_ori = dis_ori.to(self.device)
                herb_sim = herb_sim.to(self.device)
                dosage_vector = dosage_vector.to(self.device)

                judge_herb, judge_dosage = self.model_g(
                    sym_ori, syn_ori, dis_ori,
                    embeddings["Symptom"], embeddings["Syndrome"], embeddings["Disease"],
                    herb_sim, dosage_vector
                )

                loss = self.args.alpha * self.loss_fcn1(judge_herb, true_herb) + \
                       (1 - self.args.alpha) * self.loss_fcn2(judge_dosage, true_dosage)

                loss.backward()
                self.optimizer_g.step()
                training_losses += loss.item() / train_batches

            epochs.set_description("Epoch (train_Loss=%g)" % round(training_losses, 5))

        self.evaluate(test_loader, graph_data)
        os.makedirs('models', exist_ok=True)
        torch.save(self.graphEmbModel.state_dict(), 'models/graphEmbModel.pth')
        torch.save(self.model_g.state_dict(), 'models/FHPRModel.pth')
        print("Foundational herbal prescription prediction and graph representation Models saved.")
        print("=" * 60)

    def evaluate(self, test_loader, graph_data):
        """Module evaluation."""
        x = {k: v.to(self.device) for k, v in graph_data.x_dict.items()}
        edge_index = {k: v.to(self.device) for k, v in graph_data.edge_index_dict.items()}

        self.model_g.eval()
        self.graphEmbModel.eval()

        test_batches = len(test_loader)
        total_loss = 0.0
        mse_loss = mae_loss = huber_loss = 0.0
        p_5 = p_10 = p_20 = re_5 = re_10 = re_20 = 0

        for sym_ori, syn_ori, dis_ori, herb_sim, dosage_vector, true_herb, true_dosage in test_loader:
            embeddings = self.graphEmbModel(x, edge_index)
            test_herb, test_dosage = self.model_g(
                sym_ori, syn_ori, dis_ori,
                embeddings["Symptom"], embeddings["Syndrome"], embeddings["Disease"],
                herb_sim, dosage_vector
            )

            # denormalization
            test_dosage = test_dosage * self.global_dosage_range + self.global_dosage_min
            true_dosage = true_dosage * self.global_dosage_range + self.global_dosage_min

            mse_loss += self.mse(test_dosage, true_dosage).item()
            mae_loss += self.mae(test_dosage, true_dosage).item()
            huber_loss += self.huber(test_dosage, true_dosage).item()

            total_loss += (self.args.alpha * self.loss_fcn1(test_herb, true_herb) +
                           self.args.beta * self.loss_fcn2(test_dosage.squeeze(), true_dosage)).item()

            pre = sort_rows_with_indices(test_herb)
            act_result = [[idx for idx, v in enumerate(row) if v == 1]
                          for row in true_herb]

            size = sym_ori.shape[0]
            p_5 += compute_precision(pre, act_result, size, 5)
            p_10 += compute_precision(pre, act_result, size, 10)
            p_20 += compute_precision(pre, act_result, size, 20)
            re_5 += compute_recall(pre, act_result, size, 5)
            re_10 += compute_recall(pre, act_result, size, 10)
            re_20 += compute_recall(pre, act_result, size, 20)

        n = max(len(test_loader), 1)
        print('Dosage -- MSE: %.4f  MAE: %.4f  Huber: %.4f' %
              (mse_loss / n, mae_loss / n, huber_loss / n))
        print('Test loss: %.5f  P@5: %.5f  R@5: %.5f  F1@5: %.5f' %
              (total_loss / test_batches, p_5 / test_batches, re_5 / test_batches,
               2 * (p_5 / test_batches) * (re_5 / test_batches) / ((p_5 / test_batches) + (re_5 / test_batches))))
        print('P@10: %.5f  R@10: %.5f  F1@10: %.5f' %
              (p_10 / test_batches, re_10 / test_batches,
               2 * (p_10 / test_batches) * (re_10 / test_batches) / ((p_10 / test_batches) + (re_10 / test_batches))))
        print('P@20: %.5f  R@20: %.5f  F1@20: %.5f' %
              (p_20 / test_batches, re_20 / test_batches,
               2 * (p_20 / test_batches) * (re_20 / test_batches) / ((p_20 / test_batches) + (re_20 / test_batches))))



class EGHPRTrainer:
    """Stage 2: Expert-guided herbal prescription refinement module trainer"""

    def __init__(self, args, herb_numbers, sym_rule, dis_rule):
        self.args = args
        self.herb_numbers = herb_numbers
        self.sym_rule = sym_rule
        self.dis_rule = dis_rule
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_rl()

    def _setup_rl(self):
        # symptom_num + disease_num + herb_num
        state_dim = 174 + 5 + 217
        self.policy_model = PolicyNet(
            input_dim=state_dim, hidden_dim=128, num_herbs=self.herb_numbers
        ).to(self.device)
        self.value_model = ValueNet(state_dim=state_dim, hidden_dim=128).to(self.device)

        self.optimizer_r = torch.optim.Adam(
            list(self.policy_model.parameters()) +
            list(self.value_model.parameters()),
            lr=1e-4, weight_decay=1e-4
        )

        self.env = GuidelinesPrescriptionEnv(
            num_herbs=self.herb_numbers,
            sym_rule=self.sym_rule,
            dis_rule=self.dis_rule,
            max_steps=7,
            device=self.device
        )
        self.buffer = EGHPRBuffer(device=self.device)

        # RL hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.policy_epochs = 4
        self.mini_batch_size = 32
        self.value_coef = 0.5
        self.entropy_coef = 0.01

        # Early stopping
        self.best_metric = -float("inf")
        self.early_stop_patience = 10
        self.early_stop_counter = 0
        self.min_delta = 1e-4
        self.early_stop_k = 5

        # Load pre-trained foundational prescription prediction models
        graphModel_path = 'models/graphEmbModel.pth'
        fhprModel_path = 'models/FHPRModel.pth'
        if not os.path.exists(graphModel_path):
            raise FileNotFoundError(f"Model not found: {graphModel_path}. Run Stage 1 first.")
        if not os.path.exists(fhprModel_path):
            raise FileNotFoundError(f"Model not found: {fhprModel_path}. Run Stage 1 first.")

        self.graphEmbModel = TCMHeteroGNN(args=self.args).to(self.device)
        self.graphEmbModel.load_state_dict(
            torch.load(graphModel_path, map_location=self.device))
        self.model_g = FHPRModel(
            args=self.args, num_herbs=self.herb_numbers,
            sym_rule=self.sym_rule, dis_rule=self.dis_rule
        ).to(self.device)
        self.model_g.load_state_dict(
            torch.load(fhprModel_path, map_location=self.device))
        self.graphEmbModel.eval()
        self.model_g.eval()
        print("Foundational herbal prescription prediction and graph representation Models loaded.")

    def _compute_inference(self, x, edge_index, sym_ori, syn_ori, dis_ori, herb_sim, dosage_vector):
        with torch.no_grad():
            self.model_g.eval()
            self.graphEmbModel.eval()
            embeddings = self.graphEmbModel(x, edge_index)
            init_herb, init_dosage = self.model_g(
                sym_ori, syn_ori, dis_ori,
                embeddings["Symptom"], embeddings["Syndrome"], embeddings["Disease"],
                herb_sim, dosage_vector
            )
        return init_herb, init_dosage

    def train(self, train_loader, test_loader, graph_data):
        print("\n" + "=" * 60)
        print("Stage 2: Expert-guided Herbal Prescription Refinement Model Training (EGHPR)")
        print("=" * 60)

        x = {k: v.to(self.device) for k, v in graph_data.x_dict.items()}
        edge_index = {k: v.to(self.device) for k, v in graph_data.edge_index_dict.items()}

        reward_history, policy_loss_history, value_loss_history = [], [], []
        step_p5_per_epoch = []

        for epoch in range(self.args.epochs_rl):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs_rl}")
            print("-" * 40)

            step_p5_accum = [0.0] * self.env.max_steps
            step_counts = [0] * self.env.max_steps
            total_reward = 0.0
            num_batches = 0

            for batch_idx, (sym_ori, syn_ori, dis_ori, herb_sim, dosage_vector,
                            true_herb, true_dosage) in enumerate(train_loader):
                sym_ori = sym_ori.to(self.device)
                dis_ori = dis_ori.to(self.device)
                syn_ori = syn_ori.to(self.device)
                herb_sim = herb_sim.to(self.device)
                dosage_vector = dosage_vector.to(self.device)

                init_herb, _ = self._compute_inference(
                    x, edge_index, sym_ori, syn_ori, dis_ori, herb_sim, dosage_vector)

                state = self.env.reset(sym_ori, dis_ori, init_herb, true_herb)
                episode_reward = 0.0

                for step in range(self.env.max_steps):
                    prob = self.policy_model.net(state)
                    action, log_prob, _ = self.policy_model(state, deterministic=False)

                    batch_p5 = self._compute_p5_from_prob(prob, true_herb)
                    step_p5_accum[step] += batch_p5
                    step_counts[step] += 1

                    with torch.no_grad():
                        value = self.value_model(state)

                    next_state, reward, done = self.env.step(action)
                    self.buffer.store(state, action, log_prob, reward, value, done)

                    state = next_state
                    episode_reward += reward.mean().item()
                    if done:
                        break

                total_reward += episode_reward
                num_batches += 1

                if (batch_idx + 1) % 1 == 0:
                    policy_loss, value_loss = self._update_RL()
                    policy_loss_history.append(policy_loss)
                    value_loss_history.append(value_loss)

            avg_reward = total_reward / num_batches if num_batches > 0 else 0.0
            reward_history.append(avg_reward)

            if (epoch + 1) % 10 == 0:
                print("\n" + "-" * 40)
                print(f"Evaluation at Epoch {epoch + 1}")
                print("-" * 40)
                # Evaluate EGHPR
                self.evaluate(test_loader, graph_data)

            print(f"Average Reward: {avg_reward:.6f}")
            if policy_loss_history:
                recent = policy_loss_history[-10:] if len(policy_loss_history) >= 10 else policy_loss_history
                print(f"Policy Loss: {np.mean(recent):.6f}")
            if value_loss_history:
                recent = value_loss_history[-10:] if len(value_loss_history) >= 10 else value_loss_history
                print(f"Value Loss: {np.mean(recent):.6f}")

            avg_step_p5 = [step_p5_accum[s] / step_counts[s] if step_counts[s] > 0 else 0.0
                           for s in range(self.env.max_steps)]
            step_p5_per_epoch.append(avg_step_p5)


    def _compute_gae(self, rewards, values, dones, next_value):
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def _update_RL(self):
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()
        if states is None:
            return 0.0, 0.0

        T, B, state_dim = states.shape
        action_dim = actions.shape[2]

        with torch.no_grad():
            next_value = self.value_model(states[-1])

        advantages, returns = self._compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_flat = states.view(-1, state_dim)
        actions_flat = actions.view(-1, action_dim)
        old_log_probs_flat = old_log_probs.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.policy_epochs):
            indices = torch.randperm(len(states_flat))
            for start in range(0, len(states_flat), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(states_flat))
                idx = indices[start:end]

                b_states = states_flat[idx]
                b_actions = actions_flat[idx]
                b_old_log_probs = old_log_probs_flat[idx]
                b_returns = returns_flat[idx]
                b_advantages = advantages_flat[idx]

                b_new_log_probs = self.policy_model.get_log_prob(b_states, b_actions)

                b_logits = self.policy_model.net(b_states).view(-1, self.herb_numbers, 3)
                dist = torch.distributions.Categorical(logits=b_logits)
                entropy = dist.entropy().sum(dim=1).mean()

                ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                b_values = self.value_model(b_states)
                value_loss = F.mse_loss(b_values, b_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer_r.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 0.5)
                self.optimizer_r.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        self.buffer.clear()
        num_updates = self.policy_epochs * (len(states_flat) // self.mini_batch_size + 1)
        return (total_policy_loss / num_updates if num_updates > 0 else 0.0,
                total_value_loss / num_updates if num_updates > 0 else 0.0)


    #  Evaluation
    def _compute_p5_from_prob(self, prob, true_herb):
        batch_size = prob.shape[0]
        top5_indices = torch.argsort(prob, dim=1, descending=True)[:, :5]
        p5_total = 0.0
        for i in range(batch_size):
            true_set = set(torch.where(true_herb[i] > 0.5)[0].cpu().numpy())
            hits = len(set(top5_indices[i].cpu().numpy()) & true_set)
            p5_total += hits / 5.0
        return p5_total / batch_size

    def evaluate(self, test_loader, graph_data, return_metric=False, metric_k=5):
        self.policy_model.eval()
        self.model_g.eval()
        self.graphEmbModel.eval()

        x = {k: v.to(self.device) for k, v in graph_data.x_dict.items()}
        edge_index = {k: v.to(self.device) for k, v in graph_data.edge_index_dict.items()}

        base_scores = {'p5': [], 'r5': [], 'p10': [], 'r10': [], 'p20': [], 'r20': []}
        refine_scores = {'p5': [], 'r5': [], 'p10': [], 'r10': [], 'p20': [], 'r20': []}

        with torch.no_grad():
            for batch_idx, (sym_ori, syn_ori, dis_ori, herb_sim, dosage_vector,
                            true_herb, true_dosage) in enumerate(test_loader):
                if batch_idx >= 10:
                    break

                sym_ori = sym_ori.to(self.device)
                syn_ori = syn_ori.to(self.device)
                dis_ori = dis_ori.to(self.device)
                herb_sim = herb_sim.to(self.device)
                dosage_vector = dosage_vector.to(self.device)
                true_herb = true_herb.to(self.device)

                base_herb, _ = self._compute_inference(
                    x, edge_index, sym_ori, syn_ori, dis_ori, herb_sim, dosage_vector)

                state = self.env.reset(sym_ori, dis_ori, base_herb, true_herb)
                last_logits = None
                for _ in range(self.env.max_steps):
                    action, _, logits = self.policy_model(state, deterministic=True)
                    last_logits = logits
                    state, _, done = self.env.step(action)
                    if done:
                        break

                action_prob = torch.softmax(last_logits, dim=-1)
                add_prob = action_prob[:, :, 1]
                remove_prob = action_prob[:, :, 2]

                base_score = torch.sigmoid(base_herb)
                rlhf_score = base_score + 0.3 * (add_prob - remove_prob)
                rlhf_score = torch.clamp(rlhf_score, 0.0, 1.0)

                base_sorted = sort_rows_with_indices(base_score)
                rlhf_sorted = sort_rows_with_indices(rlhf_score)

                act_result = [[idx for idx, v in enumerate(row) if v == 1]
                              for row in true_herb]
                size = sym_ori.shape[0]

                for k in [5, 10, 20]:
                    bp = compute_precision(base_sorted, act_result, size, k)
                    rp = compute_precision(rlhf_sorted, act_result, size, k)
                    br = compute_recall(base_sorted, act_result, size, k)
                    rr = compute_recall(rlhf_sorted, act_result, size, k)
                    base_scores[f'p{k}'].append(bp)
                    refine_scores[f'p{k}'].append(rp)
                    base_scores[f'r{k}'].append(br)
                    refine_scores[f'r{k}'].append(rr)

        self._print_comparison(base_scores, refine_scores)

        if return_metric:
            bf1 = 2 * np.mean(base_scores[f'p{metric_k}']) * np.mean(base_scores[f'r{metric_k}']) / \
                  (np.mean(base_scores[f'p{metric_k}']) + np.mean(base_scores[f'r{metric_k}']))
            rf1 = 2 * np.mean(refine_scores[f'p{metric_k}']) * np.mean(refine_scores[f'r{metric_k}']) / \
                  (np.mean(refine_scores[f'p{metric_k}']) + np.mean(refine_scores[f'r{metric_k}']))
            return rf1 - bf1

    @staticmethod
    def _print_comparison(base, rlhf):
        print()
        for k in [5, 10, 20]:
            bp = np.mean(base[f'p{k}']); rp = np.mean(rlhf[f'p{k}'])
            br = np.mean(base[f'r{k}']); rr = np.mean(rlhf[f'r{k}'])
            bf1 = 2 * bp * br / (bp + br)
            rf1 = 2 * rp * rr / (rp + rr)
            print(f"  P@{k}: {bp:.4f} -> {rp:.4f}  |  R@{k}: {br:.4f} -> {rr:.4f}  |  F1@{k}: {bf1:.4f} -> {rf1:.4f}")

# ---------------------------------------------------------------
#  EGHPR framework setting
# ---------------------------------------------------------------
class EGHPRBuffer:
    """EGHPR buffer"""

    def __init__(self, device='cpu'):
        self.device = device
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward.detach())
        self.values.append(value.detach())
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def get(self):
        if not self.states:
            return None, None, None, None, None, None
        states_tensor = torch.stack(self.states)
        actions_tensor = torch.stack(self.actions)
        log_probs_tensor = torch.stack(self.log_probs)
        rewards_tensor = torch.stack(self.rewards)
        values_tensor = torch.stack(self.values)
        dones_tensor = torch.tensor(self.dones, device=self.device).float()
        return states_tensor, actions_tensor, log_probs_tensor, rewards_tensor, values_tensor, dones_tensor


class GuidelinesPrescriptionEnv:
    """Prescription refinement environment"""
    def __init__(self, num_herbs, sym_rule, dis_rule, max_steps=5, device='cpu'):
        self.num_herbs = num_herbs
        self.sym_rule = sym_rule
        self.dis_rule = dis_rule
        self.max_steps = max_steps
        self.device = device

    def reset(self, sym_ori, dis_ori, init_herbs, true_herb=None):
        self.sym_ori = sym_ori.to(self.device)
        self.dis_ori = dis_ori.to(self.device)
        self.true_herb = true_herb.to(self.device) if true_herb is not None else None
        self.selected_herbs = (init_herbs > 0.5).float()
        self.initial_herbs = self.selected_herbs.clone()
        self.steps = 0
        self.done = False
        self.baseline_reward = self._compute_total_reward().detach()
        return self._get_state()

    def _get_state(self):
        sym_ori = self.sym_ori.to(self.device)
        dis_ori = self.dis_ori.to(self.device)
        selected_herbs = self.selected_herbs.to(self.device)
        return torch.cat([sym_ori, dis_ori, selected_herbs], dim=1)

    def step(self, action_herb):
        action = action_herb.long().detach()
        add_mask = (action == 1)
        remove_mask = (action == 2)
        new_herbs = self.selected_herbs.clone()
        new_herbs[add_mask] = 1.0
        new_herbs[remove_mask] = 0.0
        self.selected_herbs = new_herbs
        current_reward = self._compute_total_reward()
        reward = current_reward - self.baseline_reward
        change_reward = torch.abs(self.selected_herbs - self.initial_herbs).sum(dim=1) * 0.01
        reward = reward + change_reward
        self.steps += 1
        self.done = self.steps >= self.max_steps
        return self._get_state(), reward, self.done

    def _compute_total_reward(self):
        B = self.sym_ori.size(0)
        device = self.device

        # --- Jaccard similarity  ---
        if self.true_herb is not None:
            tp = (self.selected_herbs * self.true_herb).sum(dim=1)
            pred_cnt = self.selected_herbs.sum(dim=1)
            true_cnt = self.true_herb.sum(dim=1)
            union = pred_cnt + true_cnt - tp
            similarity = tp / union.clamp(min=1)
        else:
            similarity = torch.zeros(B, device=device)

        # --- Clinical constraint  ---
        match_score = torch.zeros(B, device=device)
        n_rules = 0
        for sym_id, herb_dict in self.sym_rule.items():
            mask = (self.sym_ori[:, int(sym_id)] > 0).float()
            for herb_id, _ in herb_dict.items():
                match_score += mask * self.selected_herbs[:, int(herb_id)]
                n_rules += 1
        for dis_id, herb_dict in self.dis_rule.items():
            mask = (self.dis_ori[:, int(dis_id)] > 0).float()
            for herb_id, _ in herb_dict.items():
                match_score += mask * self.selected_herbs[:, int(herb_id)]
                n_rules += 1
        rule_match = match_score / max(n_rules, 1)

        herb_count = self.selected_herbs.sum(dim=1)
        # the minimum and maximum acceptable prescription sizes determined by the institution-specific guidelines
        N_min, N_max = 10, 30
        count_violation = (
            torch.clamp(N_min - herb_count, min=0) +
            torch.clamp(herb_count - N_max, min=0)
        ) / N_max

        constraint = rule_match - count_violation

        nu = 0.5
        beta = 0.5
        # Current reward
        reward = nu * similarity + beta * constraint
        return reward

    def get_prescription(self):
        return self.selected_herbs


class ValueNet(torch.nn.Module):
    """value network (critic)"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class PolicyNet(nn.Module):
    """policy network (actor)"""
    def __init__(self, input_dim, hidden_dim, num_herbs):
        super().__init__()
        self.num_herbs = num_herbs
        self.num_actions = 3  # 0: keep, 1: add, 2: remove
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_herbs * self.num_actions)
        )

    def forward(self, state, deterministic=False):
        logits = self.net(state)
        logits = logits.view(-1, self.num_herbs, self.num_actions)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            return action, None, logits
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)
            return action, log_prob, logits

    def get_log_prob(self, state, action):
        logits = self.net(state)
        logits = logits.view(-1, self.num_herbs, self.num_actions)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action.long()).sum(dim=1)
        return log_prob

    def get_entropy(self, state):
        logits = self.net(state)
        logits = logits.view(-1, self.num_herbs, self.num_actions)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy().sum(dim=1).mean()

def sort_rows_with_indices(data):
    """Herbs are sorted by probability"""
    results = []
    for row in data:
        indexed = [(value.item(), idx) for idx, value in enumerate(row)]
        sorted_idx = sorted(indexed, key=lambda x: x[0], reverse=True)
        results.append({
            'sorted_values': [item[0] for item in sorted_idx],
            'original_indices': [item[1] for item in sorted_idx]
        })
    return results


def compute_precision(predict_list, ground_list, size, top_k):
    """Precision@k computation"""
    relevant = 0
    total = 0
    for i in range(size):
        relevant += sum(1 for item in predict_list[i]['original_indices'][:top_k]
                        if item in ground_list[i])
        total += len(predict_list[i]['original_indices'][:top_k])
    return relevant / total


def compute_recall(predict_list, ground_list, size, top_k):
    """Recall@k computation"""
    relevant = 0
    total = 0
    for i in range(size):
        relevant += sum(1 for item in predict_list[i]['original_indices'][:top_k]
                        if item in ground_list[i])
        total += len(ground_list[i])
    return relevant / total
