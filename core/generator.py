import random
import numpy as np
from collections import defaultdict
from .vocabulary import Vocabulario
from config import CONFIG


class GeneradorBigrama:
    def __init__(self):
        self.vocabulario = Vocabulario()
        self.bigrams = self._build_language_model()

    def _build_language_model(self):
        bigrams = defaultdict(lambda: defaultdict(int))
        palabras = self.vocabulario.palabras
        for i in range(len(palabras) - 1):
            w1 = palabras[i]
            w2 = palabras[i + 1]
            bigrams[w1][w2] += 1

        for word in bigrams:
            total = sum(bigrams[word].values())
            for next_word in bigrams[word]:
                bigrams[word][next_word] /= total
        return bigrams

    def generar(self, num_palabras=None, temperatura=None, genero=None, tono=None):
        if num_palabras is None:
            num_palabras = CONFIG["max_words"]
        if temperatura is None:
            temperatura = CONFIG["temperature_default"]
        if genero is None:
            genero = "Cuento"

        word_pool = self.vocabulario.palabras
        prefixes = CONFIG["genero_prefixes"].get(genero, ["había", "en", "un"])

        texto = []
        current_word = random.choice(prefixes)
        texto.append(current_word)

        for _ in range(num_palabras - 1):
            if current_word in self.bigrams:
                candidates = self.bigrams[current_word]
                probs = np.array(list(candidates.values()))
                words = list(candidates.keys())

                if temperatura > 0:
                    probs = np.power(probs, 1.0 / temperatura)
                    probs = probs / probs.sum()
                    next_word = np.random.choice(words, p=probs)
                else:
                    next_word = max(candidates, key=candidates.get)
            else:
                next_word = random.choice(word_pool)

            if next_word == self.vocabulario.eos_token or next_word is None:
                next_word = random.choice(word_pool)

            texto.append(next_word)
            current_word = next_word

        return " ".join(texto)