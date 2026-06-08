import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from env.rl_env import EscrituraCreativaEnv
from config import CONFIG
from core.logica import LogicaPrimerOrden
from core.posproceso import concordar_frase
from core.bigram import BigramModel

_SHORTHAND = {
    "es": {
        "determinante": "det", "preposicion": "prep", "conjuncion": "conj",
        "pronombre": "pron", "verbo": "verb", "adjetivo": "adj",
        "adverbio": "adv", "sustantivo": "noun",
    },
    "en": {
        "determiner": "det", "preposition": "prep", "conjunction": "conj",
        "pronoun": "pron", "verb": "verb", "adjective": "adj",
        "adverb": "adv", "noun": "noun",
    },
}

_TRANSITIONS = {
    "__start__": {"det": 8.0, "pron": 6.0, "adv": 3.0, "noun": 2.0},
    "det":  {"noun": 8.0, "adj": 4.0},
    "prep": {"det": 6.0, "noun": 5.0, "pron": 3.0},
    "conj": {"det": 5.0, "noun": 4.0, "verb": 4.0, "pron": 3.0},
    "pron": {"verb": 8.0},
    "verb": {"det": 6.0, "prep": 4.0, "adv": 4.0, "noun": 3.0},
    "adj":  {"noun": 8.0, "conj": 3.0},
    "adv":  {"verb": 6.0, "adj": 4.0, "det": 2.0},
    "noun": {"verb": 5.0, "prep": 4.0, "adj": 4.0, "adv": 3.0, "conj": 3.0},
}

_SLOTMAP = {
    "D":"determinante","N":"sustantivo","V":"verbo",
    "P":"preposicion","J":"adjetivo","B":"adverbio",
    "C":"conjuncion","R":"pronombre",
}
_SLOTS = ("DNVPDN DNVDN DNCDNV DNVB DNJ DNPDNCDN DNVPDNJ "
          "RVPDN DNJVPB DNVPDNB BVNDNPDN DNCDNVPDN DNJCDNV "
          "VNDPDN RVPDNJ").split()

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _modelo_ruta(idioma):
    return os.path.join(MODELS_DIR, f"modelo_ppo_{idioma}.pth")


class PPOServer:
    def __init__(self, idioma="es"):
        self.idioma = idioma
        self.env = EscrituraCreativaEnv(idioma=idioma)
        self.agent = None
        self.vocabulario = self.env.vocabulario
        palabras_gen = CONFIG.get(f"palabras_por_genero_{idioma}", CONFIG["palabras_por_genero"])
        palabras_ton = CONFIG.get(f"palabras_por_tono_{idioma}", CONFIG["palabras_por_tono"])
        self.logica = LogicaPrimerOrden(
            self.vocabulario,
            palabras_gen,
            palabras_ton,
        )
        self.last_trajectory = {"observations": [], "actions": []}
        self._ps = 0
        self._sk = None
        self._seg_cat = {}
        self._entropy_coef = 0.3
        self._global_word_history = []
        self._bigram = BigramModel(self.vocabulario)
        self._init_infinitives(idioma)

    def inicializar(self):
        if self.agent is not None:
            expected_obs = self.env.observation_space.shape
            expected_act = self.env.action_space.n
            if self.agent.observation_space.shape != expected_obs or self.agent.action_space.n != expected_act:
                self.agent = None
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
            self.cargar()

    def _init_infinitives(self, idioma):
        self._inf_tokens = set()
        self._conjugated_verb_exists = False
        self._noun_genders = {}
        vocab = self.vocabulario
        for i in range(vocab.vocab_size):
            w = vocab.ind2word.get(i, '')
            cat = vocab.categoria(i)
            if cat == 'verbo' and len(w) > 2:
                if w.endswith(('ar', 'er', 'ir')):
                    self._inf_tokens.add(i)
                else:
                    self._conjugated_verb_exists = True
        # Build gender lookup from posproceso data
        from core.posproceso import _GENERO
        for w, g in _GENERO.items():
            idx = vocab.word2ind.get(w)
            if idx is not None:
                self._noun_genders[idx] = g

    def guardar(self):
        if self.agent is not None:
            torch.save(self.agent.policy.state_dict(), _modelo_ruta(self.idioma))

    def cargar(self):
        ruta = _modelo_ruta(self.idioma)
        if self.agent is not None and os.path.exists(ruta):
            state = torch.load(ruta, map_location="cpu", weights_only=True)
            self.agent.policy.load_state_dict(state, strict=False)

    def _actualizar_idioma(self, idioma):
        if idioma == self.idioma:
            return
        self.idioma = idioma
        self.vocabulario = self.env.vocabulario
        palabras_gen = CONFIG.get(f"palabras_por_genero_{idioma}", CONFIG["palabras_por_genero"])
        palabras_ton = CONFIG.get(f"palabras_por_tono_{idioma}", CONFIG["palabras_por_tono"])
        self.logica = LogicaPrimerOrden(
            self.vocabulario,
            palabras_gen,
            palabras_ton,
        )
        self._init_infinitives(idioma)

    def _syntactic_bias(self, ultimo_token, idioma):
        shorthand = _SHORTHAND.get(idioma, _SHORTHAND["es"])
        if ultimo_token is None:
            rules = _TRANSITIONS["__start__"]
        else:
            cat = self.vocabulario.categoria(ultimo_token)
            st = shorthand.get(cat, "noun")
            rules = _TRANSITIONS.get(st, {})
        bias = np.ones(self.vocabulario.vocab_size, dtype=np.float32)
        bias[self.vocabulario.eos_token] = 0.0
        bias[self.vocabulario.pad_token] = 0.0
        for i in range(self.vocabulario.vocab_size):
            if i in (self.vocabulario.eos_token, self.vocabulario.pad_token):
                continue
            cat_i = self.vocabulario.categoria(i)
            st_i = shorthand.get(cat_i, "noun")
            mult = rules.get(st_i)
            if mult is not None:
                bias[i] = mult
        return bias

    def generar(self, num_palabras=None, temperatura=None, consigna="", genero="Cuento", tono="Misterioso", idioma=None):
        if num_palabras is None:
            num_palabras = CONFIG["max_words"]
        if temperatura is None:
            temperatura = CONFIG["temperature_default"]
        if idioma is None:
            idioma = self.idioma

        self._actualizar_idioma(idioma)
        self.inicializar()

        obs, info = self.env.reset(options={"consigna": consigna, "genero": genero, "tono": tono, "idioma": idioma})
        texto_generado = []
        trajectory = {
            "observations": [obs.copy()],
            "actions": [],
            "rewards": [],
            "genero": genero,
            "tono": tono,
            "consigna": consigna,
        }

        for _ in range(num_palabras):
            action, _ = self.agent.predict(obs, deterministic=False)
            action = int(action)

            if action == self.vocabulario.eos_token:
                break
            if action == self.vocabulario.pad_token:
                continue

            palabra = self.vocabulario.ind2word[action]
            texto_generado.append(palabra)
            trajectory["actions"].append(action)

            obs, reward, terminated, truncated, info = self.env.step(action)
            trajectory["rewards"].append(reward)

            if terminated or truncated:
                break
            trajectory["observations"].append(obs.copy())

        self.last_trajectory = trajectory
        return " ".join(texto_generado)

    def generar_con_temperatura(self, num_palabras=None, temperatura=None, consigna="", genero="Cuento", tono="Misterioso", idioma=None):
        if num_palabras is None:
            num_palabras = CONFIG["max_words"]
        if temperatura is None:
            temperatura = CONFIG["temperature_default"]
        if idioma is None:
            idioma = self.idioma

        self._actualizar_idioma(idioma)
        self.inicializar()

        obs, info = self.env.reset(options={"consigna": consigna, "genero": genero, "tono": tono, "idioma": idioma})
        texto_generado = []
        trajectory = {
            "observations": [obs.copy()],
            "actions": [],
            "rewards": [],
            "genero": genero,
            "tono": tono,
            "consigna": consigna,
        }

        prompt_tokens = set(self.env._consigna_to_tokens(consigna))
        usado = {}
        ultimo_token = None
        gen_counts = {}
        last_det_idx = -1
        last_det_gender = None
        if self._sk is None:
            self._sk = _SLOTS[:]
        j = np.random.randint(len(self._sk))

        DET_GENDER_MAP = {}
        vocab = self.vocabulario
        for i in range(vocab.vocab_size):
            w = vocab.ind2word.get(i, '')
            if w in ('el', 'un', 'los', 'este', 'ese', 'aquel'):
                DET_GENDER_MAP[i] = 'm'
            elif w in ('la', 'una', 'las', 'esta', 'esa', 'aquella'):
                DET_GENDER_MAP[i] = 'f'

        FUNC = {"determinante", "preposicion", "conjuncion", "pronombre"}
        CONTENT = {"sustantivo", "verbo", "adjetivo", "adverbio"}

        BIGRAM_ALPHA = 0.35

        for paso in range(num_palabras):
            probs = self._get_action_probs(obs, temperatura)

            # Blend with bigram probabilities
            bigram_p = self._bigram.probs(ultimo_token)
            probs = (1.0 - BIGRAM_ALPHA) * probs + BIGRAM_ALPHA * bigram_p

            for i in range(len(probs)):
                if i in (self.vocabulario.eos_token, self.vocabulario.pad_token):
                    probs[i] = 0.0
                else:
                    s = self.logica.evaluar_paso(
                        token=i, genero=genero, tono=tono, prompt_tokens=prompt_tokens
                    )
                    probs[i] *= (1.0 + 0.8 * s)

            syn_bias = self._syntactic_bias(ultimo_token, idioma or self.idioma)
            probs *= syn_bias

            pk = self._sk[j]
            pc = _SLOTMAP.get(pk[paso]) if paso < len(pk) else None

            for i in range(len(probs)):
                if i in usado:
                    cat = self.vocabulario.categoria(i)
                    if cat in FUNC:
                        if usado[i] >= 1:
                            probs[i] = 0.0
                    else:
                        probs[i] = 0.0
                if pc and i not in (self.vocabulario.eos_token, self.vocabulario.pad_token):
                    if self.vocabulario.categoria(i) != pc:
                        probs[i] = 0.0

            # Global frequency penalty: reduce probability of recently over-used words
            if self._global_word_history:
                recent_counts = {}
                for a in self._global_word_history[-500:]:
                    recent_counts[a] = recent_counts.get(a, 0) + 1
                for i in range(len(probs)):
                    cnt = recent_counts.get(i, 0)
                    if cnt > 2:
                        probs[i] /= (1.0 + 0.5 * (cnt - 2))

            # Phase 2: Penalizar infinitivos si el slot es VERBO
            if pc == 'verbo' and self._conjugated_verb_exists:
                for i in self._inf_tokens:
                    if i < len(probs):
                        probs[i] *= 0.15

            # Phase 3: Decaimiento por repetición para palabras de contenido
            for i, cnt in gen_counts.items():
                if i < len(probs):
                    cat = self.vocabulario.categoria(i)
                    if cat in CONTENT:
                        probs[i] /= (1.0 + 0.8 * cnt)

            # Phase 3: Seg_cat diversity (last 5 per category)
            if pc:
                for idx in self._seg_cat.get(pc, []):
                    if idx in range(len(probs)):
                        probs[idx] = 0.0

            # Phase 4: DET-N gender agreement bias
            if pc == 'sustantivo' and last_det_gender is not None:
                for i in range(len(probs)):
                    g = self._noun_genders.get(i)
                    if g and g != last_det_gender:
                        probs[i] *= 0.3
            if pc == 'determinante' and last_det_idx >= 0:
                for i in range(len(probs)):
                    g = DET_GENDER_MAP.get(i)
                    noun_g = self._noun_genders.get(last_det_idx)
                    if g and noun_g and g != noun_g:
                        probs[i] *= 0.3

            if probs.sum() == 0.0:
                probs[:] = 1.0
                if pc:
                    for i in range(len(probs)):
                        if self.vocabulario.categoria(i) != pc:
                            probs[i] = 0.0
                probs[self.vocabulario.eos_token] = 0.0
                probs[self.vocabulario.pad_token] = 0.0
            probs /= probs.sum()

            if temperatura > 0:
                probs = np.power(probs, 1.0 / temperatura)
                if probs.sum() == 0.0:
                    probs[:] = 1.0
                    probs[self.vocabulario.eos_token] = 0.0
                    probs[self.vocabulario.pad_token] = 0.0
                else:
                    # Top‑k: zero out everything below the 25th highest prob
                    k = min(25, len(probs))
                    threshold = -np.sort(-probs)[k - 1] if k > 0 else 0.0
                    probs[probs < threshold] = 0.0
                probs /= probs.sum()
                action = np.random.choice(len(probs), p=probs)
            else:
                action = np.argmax(probs)

            if action == self.vocabulario.eos_token:
                break

            usado[action] = 1 + usado.get(action, 0)
            gen_counts[action] = gen_counts.get(action, 0) + 1
            ultimo_token = action
            palabra = self.vocabulario.ind2word[action]
            texto_generado.append(palabra)
            trajectory["actions"].append(action)
            act_cat = self.vocabulario.categoria(action)
            if act_cat == 'determinante':
                last_det_idx = action
                last_det_gender = DET_GENDER_MAP.get(action)

            obs, reward, terminated, truncated, info = self.env.step(action)
            trajectory["rewards"].append(reward)

            if terminated or truncated:
                break
            trajectory["observations"].append(obs.copy())

        self.last_trajectory = trajectory
        self._ps += 1
        for act in trajectory["actions"]:
            self._global_word_history.append(act)
            if len(self._global_word_history) > 2000:
                self._global_word_history = self._global_word_history[-1000:]
            cat = self.vocabulario.categoria(act)
            if cat not in self._seg_cat:
                self._seg_cat[cat] = []
            self._seg_cat[cat].append(act)
            if len(self._seg_cat[cat]) > 5:
                self._seg_cat[cat].pop(0)
        texto = " ".join(texto_generado)
        texto = concordar_frase(texto, self.vocabulario)
        return texto

    def _get_action_probs(self, obs, temperatura, forbid_tokens=None):
        if self.agent is None:
            return np.ones(self.vocabulario.vocab_size) / self.vocabulario.vocab_size

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            features = self.agent.policy.extract_features(obs_tensor)
            latent_pi, _ = self.agent.policy.mlp_extractor(features)
            logits = self.agent.policy.action_net(latent_pi)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        if forbid_tokens:
            probs[list(forbid_tokens)] = 0.0
            probs /= probs.sum()

        return probs

    def entrenar(self, trajectory, reward_escalar):
        if self.agent is None or not trajectory["observations"] or not trajectory["actions"]:
            return

        obs_array = np.array(trajectory["observations"], dtype=np.float32)
        action_array = np.array(trajectory["actions"], dtype=np.int64)
        per_step = np.array(trajectory.get("rewards", []), dtype=np.float32)

        n = len(action_array)
        obs_tensor = torch.from_numpy(obs_array[:n])
        action_tensor = torch.from_numpy(action_array)

        if len(per_step) == n and reward_escalar > 0:
            per_word = per_step.copy()
        elif len(per_step) == n and reward_escalar < 0:
            per_word = -(1.0 - per_step)
        else:
            per_word = np.full(n, reward_escalar, dtype=np.float32)

        unique, counts = np.unique(action_array, return_counts=True)
        for palabra_id in unique[counts > 2]:
            seen = 0
            for i in range(n):
                if action_array[i] == palabra_id:
                    seen += 1
                    if seen > 2:
                        per_word[i] = 0.0

        adv = per_word.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = torch.from_numpy(adv)

        gamma = CONFIG["ppo_config"]["gamma"]
        returns = np.zeros(n, dtype=np.float32)
        G = 0.0
        for i in range(n - 1, -1, -1):
            G = per_word[i] + gamma * G
            returns[i] = G
        returns_t = torch.from_numpy(returns)

        genero = trajectory.get("genero", "Cuento")
        tono = trajectory.get("tono", "Misterioso")
        consigna = trajectory.get("consigna", "")
        prompt_tokens = set(self.env._consigna_to_tokens(consigna))

        for _ in range(3):
            features = self.agent.policy.extract_features(obs_tensor)
            latent_pi, latent_vf = self.agent.policy.mlp_extractor(features)
            logits = self.agent.policy.action_net(latent_pi)
            values = self.agent.policy.value_net(latent_vf)

            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(action_tensor)

            loss_ppo = -(log_probs * adv_t).mean()

            value_loss = torch.nn.functional.mse_loss(
                values.squeeze(), returns_t
            )

            loss_logica = self.logica.loss(
                log_probs, action_array, genero, tono, prompt_tokens,
                lambda_semantica=CONFIG["lambda_semantica"],
                lambda_estructura=CONFIG["lambda_estructura"],
            )

            entropy = dist.entropy().mean()
            loss = loss_ppo + 0.5 * value_loss + loss_logica - self._entropy_coef * entropy

            self.agent.policy.optimizer.zero_grad()
            loss.backward()
            self.agent.policy.optimizer.step()

        self._entropy_coef *= 0.995
        self._entropy_coef = max(self._entropy_coef, 0.01)
        trajectory["rewards"] = []
        self.guardar()