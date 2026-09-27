import numpy as np
import torch
import torch.nn as nn


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(64, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class _PolicyWrapper:
    """Same interface as tl_ai/evaluate.py's _PolicyWrapper, trimmed to
    only what battle_engine.py calls (action_probs; greedy_action kept
    too since it's nearly free and a natural thing to want later)."""

    def __init__(self, net):
        self.net = net

    def _raw_logits(self, obs):
        with torch.no_grad():
            logits_t, _ = self.net.forward(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return logits_t.squeeze(0).cpu().numpy().astype(np.float64)

    def greedy_action(self, obs, mask):
        logits = self._raw_logits(obs)
        masked = np.where(mask, logits, -1e9)
        return int(np.argmax(masked))

    def action_probs(self, obs, mask):
        logits = self._raw_logits(obs)
        masked = np.where(mask, logits, -1e9)
        shifted = masked - np.max(masked)
        probs = np.exp(shifted)
        probs /= probs.sum()
        return probs


def _infer_hidden(state_dict):
    """Reconstructs the hidden-layer tuple from the checkpoint's own
    tensor shapes -- see tl_ai/evaluate.py's _infer_torch_hidden for the
    full rationale (this project's models vary in hidden-layer depth/
    width across checkpoints, so this must be read from the file, not
    assumed)."""
    hidden = []
    i = 0
    while f"trunk.{i}.weight" in state_dict:
        hidden.append(state_dict[f"trunk.{i}.weight"].shape[0])
        i += 2
    return tuple(hidden)


def load_policy(model_path, obs_dim, action_dim, hidden=None):
    state_dict = torch.load(model_path, map_location="cpu")
    inferred = _infer_hidden(state_dict)
    net = ActorCriticNet(obs_dim, action_dim, hidden=hidden or inferred)
    net.load_state_dict(state_dict)
    net.eval()
    return _PolicyWrapper(net)
