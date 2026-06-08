import numpy as np
import torch


class LogicaPrimerOrden:
    def __init__(self, vocabulario, palabras_por_genero, palabras_por_tono):
        self.vocab = vocabulario
        self._palabras_por_genero = {
            k: set(v) for k, v in palabras_por_genero.items()
        }
        self._palabras_por_tono = {
            k: set(v) for k, v in palabras_por_tono.items()
        }

    @staticmethod
    def negacion(a):
        return 1.0 - a

    @staticmethod
    def conjuncion(a, b):
        return max(0.0, a + b - 1.0)

    @staticmethod
    def disyuncion(a, b):
        return min(1.0, a + b)

    @staticmethod
    def implicacion(a, b):
        return min(1.0, 1.0 - a + b)

    def es_del_genero(self, token, genero):
        palabra = self.vocab.ind2word.get(token, "")
        return 1.0 if palabra in self._palabras_por_genero.get(genero, set()) else 0.0

    def es_del_tono(self, token, tono):
        palabra = self.vocab.ind2word.get(token, "")
        return 1.0 if palabra in self._palabras_por_tono.get(tono, set()) else 0.0

    def es_relevante(self, token, prompt_tokens):
        return 1.0 if token in prompt_tokens else 0.0

    def es_categoria(self, token, categoria):
        return 1.0 if self.vocab.categoria(token) == categoria else 0.0

    def es_sustantivo(self, token):
        return self.es_categoria(token, "sustantivo")

    def es_verbo(self, token):
        return self.es_categoria(token, "verbo")

    def es_adjetivo(self, token):
        return self.es_categoria(token, "adjetivo")

    def es_determinante(self, token):
        return self.es_categoria(token, "determinante")

    def es_preposicion(self, token):
        return self.es_categoria(token, "preposicion")

    def es_pronombre(self, token):
        return self.es_categoria(token, "pronombre")

    def es_conjuncion(self, token):
        return self.es_categoria(token, "conjuncion")

    def es_adverbio(self, token):
        return self.es_categoria(token, "adverbio")

    def es_conjugado(self, token):
        palabra = self.vocab.ind2word.get(token, "")
        cat = self.vocab.categoria(token)
        if cat != "verbo":
            return False
        if not palabra or len(palabra) <= 2:
            return True
        return not palabra.endswith(('ar', 'er', 'ir'))

    # ------------------------------------------------------------------
    # Evaluación semántica (por paso)
    #
    #   score(x) = es_del_genero(x) ∨ es_del_tono(x) ∨ es_relevante(x)
    # ------------------------------------------------------------------
    def evaluar_paso(self, token, genero, tono, prompt_tokens):
        p_gen = self.es_del_genero(token, genero)
        p_tono = self.es_del_tono(token, tono)
        p_rel = self.es_relevante(token, prompt_tokens)
        score = 0.7 * max(p_gen, p_tono) + 0.3 * p_rel
        if score == 0.0:
            cat = self.vocab.categoria(token)
            if cat in ("determinante", "preposicion", "conjuncion", "pronombre"):
                score = 0.2
        if self.es_conjugado(token):
            score += 0.4
            score = min(score, 1.0)
        return score

    def evaluar_trayectoria_semantica(self, acciones, genero, tono, prompt_tokens):
        return np.array([
            self.evaluar_paso(t, genero, tono, prompt_tokens)
            for t in acciones
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Evaluación estructural (sintaxis — trayectoria completa)
    #
    # Reglas (Łukasiewicz):
    #
    #   R1: ∀i (det(i) → ∨_{j∈[i+1,i+3]} sust(j))
    #   R2: ∀i (prep(i) → ∨_{j∈[i+1,i+4]} (sust(j) ∨ det(j)))
    #   R3: ∀i  ∨_{j∈[i-2,i+2]} verbo(j)
    #   R4: ∀i (adj(i) → ∨_{j∈[i-2,i+2], j≠i} sust(j))
    #   R5: ∀i (pron(i) → verbo(i+1))
    #
    # Para cada posición i se promedian las reglas que aplican.
    # Si ninguna aplica → 0.5 (neutro).
    # ------------------------------------------------------------------
    def evaluar_estructura(self, acciones):
        n = len(acciones)
        scores = np.zeros(n, dtype=np.float32)
        if n < 2:
            return scores

        cats = [self.vocab.categoria(t) for t in acciones]

        for i in range(n):
            reglas = []

            # R1: Det → Sust dentro de 3 posiciones
            if cats[i] == "determinante":
                window = cats[i + 1 : min(i + 4, n)]
                reglas.append(1.0 if any(c == "sustantivo" for c in window) else 0.0)

            # R2: Prep → Sust/Det dentro de 4 posiciones
            if cats[i] == "preposicion":
                window = cats[i + 1 : min(i + 5, n)]
                reglas.append(
                    1.0 if any(c in ("sustantivo", "determinante") for c in window) else 0.0
                )

            # R3: Verbo en ventana de ±2 alrededor de cada palabra
            left = max(0, i - 2)
            right = min(n, i + 3)
            window = cats[left:right]
            reglas.append(1.0 if any(c == "verbo" for c in window) else 0.0)

            # R4: Adj cerca de Sust (±2)
            if cats[i] == "adjetivo":
                left = max(0, i - 2)
                right = min(n, i + 3)
                cerca = any(
                    cats[j] == "sustantivo"
                    for j in range(left, right)
                    if j != i
                )
                reglas.append(1.0 if cerca else 0.0)

            # R5: Pron → Verbo en la siguiente posición
            if cats[i] == "pronombre" and i + 1 < n:
                reglas.append(1.0 if cats[i + 1] == "verbo" else 0.0)

            if reglas:
                scores[i] = sum(reglas) / len(reglas)

        return scores

    # ------------------------------------------------------------------
    # Loss combinada: semántica + estructural
    # ------------------------------------------------------------------
    def loss(self, log_probs, acciones, genero, tono, prompt_tokens,
             lambda_semantica=0.3, lambda_estructura=0.2):
        if len(acciones) == 0:
            return torch.tensor(0.0, requires_grad=True)

        rep_penalty = np.ones(len(acciones), dtype=np.float32)
        counts = {}
        for i, t in enumerate(int(a) for a in acciones):
            counts[t] = counts.get(t, 0) + 1
            if counts[t] > 3:
                rep_penalty[i] = 0.0

        scores_sem = self.evaluar_trayectoria_semantica(
            acciones, genero, tono, prompt_tokens
        )
        scores_sem = scores_sem * rep_penalty
        scores_sem_t = torch.from_numpy(scores_sem)
        loss_sem = -(log_probs * scores_sem_t).mean()

        scores_est = self.evaluar_estructura(acciones)
        scores_est = scores_est * rep_penalty
        scores_est_t = torch.from_numpy(scores_est)
        loss_est = -(log_probs * scores_est_t).mean()

        return lambda_semantica * loss_sem + lambda_estructura * loss_est
