import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from env.rl_env import EscrituraCreativaEnv
from config import CONFIG


class PPOServer:
    def __init__(self):
        self.env = EscrituraCreativaEnv()
        self.agent = None
        self.vocabulario = self.env.vocabulario

    def inicializar(self):
        if self.agent is None:
            self.agent = PPO(
                ActorCriticPolicy,
                self.env,
                learning_rate=CONFIG["ppo_config"]["learning_rate"],
                n_steps=CONFIG["ppo_config"]["n_steps"],
                batch_size=CONFIG["ppo_config"]["batch_size"],
                n_epochs=CONFIG["ppo_config"]["n_epochs"],
                gamma=CONFIG["ppo_config"]["gamma"],
                verbose=0
            )

    def generar(self, num_palabras=None, temperatura=None):
        if num_palabras is None:
            num_palabras = CONFIG["max_words"]
        if temperatura is None:
            temperatura = CONFIG["temperature_default"]

        if self.agent is None:
            self.inicializar()

        obs, info = self.env.reset()
        texto_generado = []

        for _ in range(num_palabras):
            action, _ = self.agent.predict(obs, deterministic=False)

            if action == self.vocabulario.eos_token:
                break

            palabra = self.vocabulario.ind2word.get(action, "<unk>")
            texto_generado.append(palabra)

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                break

        return " ".join(texto_generado)

    def generar_con_temperatura(self, num_palabras=None, temperatura=None):
        if num_palabras is None:
            num_palabras = CONFIG["max_words"]
        if temperatura is None:
            temperatura = CONFIG["temperature_default"]

        if self.agent is None:
            self.inicializar()

        obs, info = self.env.reset()
        texto_generado = []

        for _ in range(num_palabras):
            action_probs = self._get_action_probs(obs, temperatura)

            if temperatura == 0:
                action = np.argmax(action_probs)
            else:
                probs = action_probs / np.sum(action_probs)
                probs = np.power(probs, 1.0 / temperatura)
                probs = probs / np.sum(probs)
                action = np.random.choice(len(probs), p=probs)

            if action == self.vocabulario.eos_token:
                break

            palabra = self.vocabulario.ind2word.get(action, "<unk>")
            texto_generado.append(palabra)

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                break

        return " ".join(texto_generado)

    def _get_action_probs(self, obs, temperatura):
        if self.agent is None:
            return np.ones(self.vocabulario.vocab_size) / self.vocabulario.vocab_size

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0)
            logits = self.agent.policy.actor(obs_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        return probs