import random
import numpy as np
from config import CONFIG


TEMPLATES = [
    ["DET", "SUST", "VERBO", "PREP", "DET", "SUST"],
    ["DET", "SUST", "VERBO", "DET", "SUST"],
    ["DET", "SUST", "CONJ", "DET", "SUST", "VERBO"],
    ["DET", "SUST", "VERBO", "ADV"],
    ["DET", "SUST", "ADJ"],
    ["DET", "SUST", "PREP", "DET", "SUST", "CONJ", "DET", "SUST"],
    ["DET", "SUST", "VERBO", "PREP", "DET", "SUST", "ADJ"],
    ["PRON", "VERBO", "PREP", "DET", "SUST"],
    ["DET", "SUST", "ADJ", "VERBO", "PREP", "ADV"],
    ["DET", "SUST", "VERBO", "PREP", "DET", "SUST", "ADV"],
    ["ADV", "VERBO", "DET", "SUST", "PREP", "DET", "SUST"],
    ["DET", "SUST", "CONJ", "DET", "SUST", "VERBO", "PREP", "DET", "SUST"],
    ["DET", "SUST", "ADJ", "CONJ", "DET", "SUST", "VERBO"],
    ["VERBO", "DET", "SUST", "PREP", "DET", "SUST"],
    ["PRON", "VERBO", "DET", "SUST", "ADJ"],
]


def _get_word_pools(vocab):
    pools = {
        "DET": [], "SUST": [], "VERBO": [], "PREP": [],
        "ADJ": [], "ADV": [], "CONJ": [], "PRON": [],
    }
    for token in range(vocab.vocab_size):
        word = vocab.ind2word.get(token)
        if not word:
            continue
        cat = vocab.categoria(token)
        if cat == "determinante": pools["DET"].append(token)
        elif cat == "sustantivo": pools["SUST"].append(token)
        elif cat == "verbo": pools["VERBO"].append(token)
        elif cat == "preposicion": pools["PREP"].append(token)
        elif cat == "adjetivo": pools["ADJ"].append(token)
        elif cat == "adverbio": pools["ADV"].append(token)
        elif cat == "conjuncion": pools["CONJ"].append(token)
        elif cat == "pronombre": pools["PRON"].append(token)

    # Excluir tokens especiales
    for cat in pools:
        pools[cat] = [t for t in pools[cat]
                      if t not in (vocab.pad_token, vocab.eos_token)]
    return pools


def _get_combo_pools(vocab, genero, tono):
    palabras_gen = set(CONFIG["palabras_por_genero"].get(genero, []))
    palabras_ton = set(CONFIG["palabras_por_tono"].get(tono, []))
    todas = palabras_gen | palabras_ton

    pools = {"SUST": [], "VERBO": [], "ADJ": [], "ADV": [], "CONJ": []}
    for palabra in todas:
        token = vocab.word2ind.get(palabra)
        if token is None:
            continue
        cat = vocab.categoria(token)
        cat_key = {
            "determinante": None, "preposicion": None,
            "conjuncion": "CONJ", "pronombre": None,
            "sustantivo": "SUST", "verbo": "VERBO",
            "adjetivo": "ADJ", "adverbio": "ADV",
        }.get(cat)
        if cat_key is not None:
            pools[cat_key].append(token)

    # Si algun pool esta vacio, usar pool general
    for k in pools:
        if not pools[k]:
            pools[k] = None
    return pools


class DatasetOraciones:
    def __init__(self, vocabulario, prompt_max=None, context_size=None,
                 variantes=500):
        self.vocab = vocabulario
        self.prompt_max = prompt_max or CONFIG["prompt_max_tokens"]
        self.context_size = context_size or CONFIG["context_size"]
        self.pools = _get_word_pools(vocabulario)
        self.generos_list = CONFIG["generos"]
        self.tonos_list = CONFIG["tonos"]

        print(f"  Pools: DET={len(self.pools['DET'])} SUST={len(self.pools['SUST'])} "
              f"VERBO={len(self.pools['VERBO'])} PREP={len(self.pools['PREP'])} "
              f"ADJ={len(self.pools['ADJ'])} ADV={len(self.pools['ADV'])} "
              f"CONJ={len(self.pools['CONJ'])} PRON={len(self.pools['PRON'])}")

        self.ejemplos = []
        combos = [(g, t) for g in CONFIG["generos"] for t in CONFIG["tonos"]]

        for _ in range(variantes):
            for genero, tono in combos:
                template = random.choice(TEMPLATES)
                combo_pools = _get_combo_pools(vocabulario, genero, tono)
                tokens = self._generar(template, combo_pools)
                if tokens and len(tokens) >= 2:
                    self.ejemplos.append({
                        "tokens": tokens,
                        "genero": genero,
                        "tono": tono,
                    })

        random.shuffle(self.ejemplos)

    def _generar(self, template, combo_pools):
        tokens = []
        for slot in template:
            if slot in self.pools:
                pool = self.pools[slot]
                if slot in combo_pools and combo_pools[slot] and random.random() < 0.7:
                    pool = combo_pools[slot]
                if not pool:
                    return None
                token = random.choice(pool)
                tokens.append(token)
            else:
                return None
        return tokens

    def __len__(self):
        return len(self.ejemplos)

    def _onehot(self, value, options):
        arr = np.zeros(len(options), dtype=np.int64)
        if value in options:
            arr[options.index(value)] = 1
        return arr

    def generar_obs(self, ejemplo, pos):
        tokens = ejemplo["tokens"]
        genero = ejemplo["genero"]
        tono = ejemplo["tono"]

        genero_oh = self._onehot(genero, self.generos_list)
        tono_oh = self._onehot(tono, self.tonos_list)

        context = tokens[max(0, pos - self.context_size):pos]
        if len(context) < self.context_size:
            context = [self.vocab.pad_token] * (self.context_size - len(context)) + context

        prompt = [self.vocab.pad_token] * self.prompt_max

        return np.concatenate([
            genero_oh,
            tono_oh,
            np.array(prompt, dtype=np.int64),
            np.array(context, dtype=np.int64),
        ])

    def generar_lotes(self, batch_size=32):
        indices = list(range(len(self.ejemplos)))
        np.random.shuffle(indices)

        for idx in indices:
            ejemplo = self.ejemplos[idx]
            tokens = ejemplo["tokens"]
            for pos in range(1, len(tokens)):
                obs = self.generar_obs(ejemplo, pos)
                target = tokens[pos]
                yield obs, target
