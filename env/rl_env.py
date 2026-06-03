import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.vocabulary import Vocabulario
from core.logica import LogicaPrimerOrden
from config import CONFIG


class EscrituraCreativaEnv(gym.Env):
    metadata = {"render_modes": ["text"]}

    def __init__(self, max_length=None, context_size=None, idioma="es"):
        super().__init__()
        self.idioma = idioma
        self.vocabulario = Vocabulario(idioma=idioma)
        palabras_gen = CONFIG.get(f"palabras_por_genero_{idioma}", CONFIG["palabras_por_genero"])
        palabras_ton = CONFIG.get(f"palabras_por_tono_{idioma}", CONFIG["palabras_por_tono"])
        self.logica = LogicaPrimerOrden(
            self.vocabulario,
            palabras_gen,
            palabras_ton,
        )
        self.max_length = max_length or CONFIG["max_words"]
        self.context_size = context_size or CONFIG["context_size"]
        self.prompt_max = CONFIG["prompt_max_tokens"]

        self.action_space = spaces.Discrete(self.vocabulario.vocab_size)

        self._obs_dim = 3 + 3 + self.prompt_max + self.context_size
        self.observation_space = spaces.Box(
            low=0,
            high=self.vocabulario.vocab_size,
            shape=(self._obs_dim,),
            dtype=np.int64
        )

        self.context = [self.vocabulario.pad_token] * self.context_size
        self.generated_tokens = []
        self.genero_onehot = np.zeros(3, dtype=np.int64)
        self.tono_onehot = np.zeros(3, dtype=np.int64)
        self.prompt_tokens = [self.vocabulario.pad_token] * self.prompt_max

    def _onehot(self, value, options):
        arr = np.zeros(len(options), dtype=np.int64)
        if value in options:
            arr[options.index(value)] = 1
        return arr

    def _consigna_to_tokens(self, text):
        tokens = []
        for word in text.lower().split():
            idx = self.vocabulario.word2ind.get(word, self.vocabulario.pad_token)
            tokens.append(idx)
            if len(tokens) >= self.prompt_max:
                break
        tokens += [self.vocabulario.pad_token] * (self.prompt_max - len(tokens))
        return tokens

    def _get_obs(self):
        return np.concatenate([
            self.genero_onehot,
            self.tono_onehot,
            np.array(self.prompt_tokens, dtype=np.int64),
            np.array(self.context, dtype=np.int64),
        ])

    def _get_info(self):
        return {
            "text": " ".join([
                self.vocabulario.ind2word.get(t, "<unk>")
                for t in self.generated_tokens
            ])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        genero = opts.get("genero", "Cuento")
        tono = opts.get("tono", "Misterioso")
        consigna = opts.get("consigna", "")
        idioma = opts.get("idioma", self.idioma)

        self.idioma = idioma
        self.vocabulario = Vocabulario(idioma=idioma)
        palabras_gen = CONFIG.get(f"palabras_por_genero_{idioma}", CONFIG["palabras_por_genero"])
        palabras_ton = CONFIG.get(f"palabras_por_tono_{idioma}", CONFIG["palabras_por_tono"])
        self.logica = LogicaPrimerOrden(
            self.vocabulario,
            palabras_gen,
            palabras_ton,
        )
        self.action_space = spaces.Discrete(self.vocabulario.vocab_size)
        self.observation_space = spaces.Box(
            low=0,
            high=self.vocabulario.vocab_size,
            shape=(self._obs_dim,),
            dtype=np.int64
        )
        self.generos_list = CONFIG.get(f"generos_{idioma}", CONFIG["generos"])
        self.tonos_list = CONFIG.get(f"tonos_{idioma}", CONFIG["tonos"])
        self.genero = genero
        self.tono = tono
        self.genero_onehot = self._onehot(genero, self.generos_list)
        self.tono_onehot = self._onehot(tono, self.tonos_list)
        self.prompt_tokens = self._consigna_to_tokens(consigna)
        self.context = [self.vocabulario.pad_token] * self.context_size
        self.generated_tokens = []
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action == self.vocabulario.eos_token:
            terminated = True
            truncated = False
            reward = 0.0
        elif action == self.vocabulario.pad_token:
            terminated = False
            truncated = False
            reward = 0.0
        elif len(self.generated_tokens) >= self.max_length:
            terminated = False
            truncated = True
            reward = 0.0
        else:
            self.generated_tokens.append(action)
            self.context = self.context[1:] + [action]
            if len(self.generated_tokens) >= self.max_length:
                terminated = False
                truncated = True
            else:
                terminated = False
                truncated = False
            reward = float(self.logica.evaluar_paso(
                action, self.genero, self.tono, set(self.prompt_tokens)
            ))

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        return " ".join([
            self.vocabulario.ind2word.get(t, "<unk>")
            for t in self.generated_tokens
        ])