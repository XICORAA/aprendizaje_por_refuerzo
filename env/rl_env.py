import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.vocabulary import Vocabulario
from config import CONFIG


class EscrituraCreativaEnv(gym.Env):
    metadata = {"render_modes": ["text"]}

    def __init__(self, max_length=None, context_size=None):
        super().__init__()
        self.vocabulario = Vocabulario()
        self.max_length = max_length or CONFIG["max_words"]
        self.context_size = context_size or CONFIG["context_size"]

        self.action_space = spaces.Discrete(self.vocabulario.vocab_size)
        self.observation_space = spaces.Box(
            low=0,
            high=self.vocabulario.vocab_size,
            shape=(self.context_size,),
            dtype=np.int64
        )

        self.current_text = []
        self.context = [self.vocabulario.pad_token] * self.context_size
        self.generated_tokens = []

    def _get_obs(self):
        return np.array(self.context, dtype=np.int64)

    def _get_info(self):
        return {
            "text": " ".join([
                self.vocabulario.ind2word.get(t, "<unk>")
                for t in self.generated_tokens
            ])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_text = []
        self.context = [self.vocabulario.pad_token] * self.context_size
        self.generated_tokens = []
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action == self.vocabulario.eos_token or len(self.generated_tokens) >= self.max_length:
            terminated = True
            truncated = False
            reward = 0.0
        else:
            self.generated_tokens.append(action)
            self.context = self.context[1:] + [action]
            terminated = len(self.generated_tokens) >= self.max_length
            truncated = False
            reward = 0.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        return " ".join([
            self.vocabulario.ind2word.get(t, "<unk>")
            for t in self.generated_tokens
        ])