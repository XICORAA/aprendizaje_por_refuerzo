import json
import os


def guardar_log_recompensa(consigna, genero, tono, temperatura, texto, recompensa):
    log_entry = {
        "consigna": consigna,
        "genero": genero,
        "tono": tono,
        "temperatura": temperatura,
        "texto": texto,
        "recompensa": recompensa
    }

    filename = "logs_alineacion.json"

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    return True