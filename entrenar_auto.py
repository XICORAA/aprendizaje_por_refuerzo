import time
import random
import numpy as np
from rl.agent import PPOServer


def parsear_ejemplos(ruta="README.md"):
    import re
    ejemplos = []
    genero_actual = None
    tono_actual = None
    with open(ruta, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"###\s+(.+?)\s*/\s*(.+)", line)
            if m:
                genero_actual = m.group(1).strip()
                tono_actual = m.group(2).strip()
                continue
            if line.startswith("| `"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    consigna = parts[1].strip("`").strip()
                    ejemplos.append((consigna, genero_actual, tono_actual))
    return ejemplos


def entrenar():
    ejemplos = parsear_ejemplos()
    print(f"  {len(ejemplos)} ejemplos")

    server = PPOServer(idioma="es")
    server.inicializar()
    print(f"  Vocab size: {server.vocabulario.vocab_size}")

    NUM_EPOCAS = 12
    LOTE = 5

    for epoca in range(1, NUM_EPOCAS + 1):
        random.shuffle(ejemplos)
        total = 0
        likes = 0
        acum = []

        for consigna, genero, tono in ejemplos:
            texto = server.generar_con_temperatura(
                consigna=consigna, genero=genero, tono=tono, temperatura=0.6,
            )

            t = server.last_trajectory
            if not t["actions"] or len(t["actions"]) < 3:
                continue

            acciones = t["actions"]
            per_step = np.array(t.get("rewards", []), dtype=np.float32)
            sem_mean = float(np.mean(per_step)) if len(per_step) > 0 else 0.0

            est_scores = server.logica.evaluar_estructura(acciones)

            cats = {"determinante": 0, "preposicion": 0, "conjuncion": 0,
                    "pronombre": 0, "verbo": 0, "adjetivo": 0, "adverbio": 0, "sustantivo": 0}
            for a in acciones:
                c = server.vocabulario.categoria(a)
                if c in cats:
                    cats[c] += 1
            cat_count = sum(1 for c in ["determinante", "preposicion", "verbo", "sustantivo"] if cats[c] > 0)
            cat_coverage = cat_count / 4.0

            unicas = len(set(acciones))
            total_palabras = len(acciones)
            diversidad = unicas / max(total_palabras, 1)

            est_mean = float(np.mean(est_scores)) if len(est_scores) > 0 else 0.0
            score = 0.20 * sem_mean + 0.30 * est_mean + 0.30 * cat_coverage + 0.20 * diversidad
            penalty = 0.0
            if cats.get("determinante", 0) == 0: penalty -= 0.4
            if cats.get("preposicion", 0) == 0: penalty -= 0.3
            if cats.get("conjuncion", 0) == 0: penalty -= 0.1
            reward = max(-1.0, min(1.0, score + penalty))

            for i, a in enumerate(acciones):
                cat = server.vocabulario.categoria(a)
                if cat in ("determinante", "preposicion", "conjuncion", "pronombre"):
                    per_step[i] = max(per_step[i], 0.6)
            est_per_word = np.full(len(acciones), est_mean, dtype=np.float32)
            combined_per_step = 0.5 * per_step + 0.5 * est_per_word
            t["rewards"] = combined_per_step.tolist()

            t["genero"] = genero
            t["tono"] = tono
            t["consigna"] = consigna
            t["_sem_mean"] = sem_mean

            acum.append((t, reward))
            if reward > 0:
                likes += 1
            total += 1

            if len(acum) >= LOTE:
                for tr, rw in acum:
                    server.entrenar(tr, rw)
                acum = []

        if acum:
            for tr, rw in acum:
                server.entrenar(tr, rw)

        server.guardar()

        # Print sample
        if epoca % 4 == 0 or epoca == 1:
            s = server.generar_con_temperatura(
                consigna=ejemplos[0][0], genero=ejemplos[0][1], tono=ejemplos[0][2], temperatura=0.5
            )
            print(f"E{epoca:2d}/{NUM_EPOCAS} L {likes:3d}/{total}  ej: {s[:60]}")

    print(f"\n  COMPLETO ({NUM_EPOCAS} epocas)")


if __name__ == "__main__":
    t0 = time.time()
    entrenar()
    t1 = time.time()
    print(f"  Tiempo: {t1 - t0:.0f}s")
