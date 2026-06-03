import json
import os


LOGS_FILE = "logs/logs_alineacion.json"


def guardar_log_recompensa(consigna, genero, tono, temperatura, texto, recompensa):
    logs = []
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append({
        "consigna": consigna,
        "genero": genero,
        "tono": tono,
        "temperatura": temperatura,
        "texto": texto,
        "recompensa": recompensa,
    })

    with open(LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    return True
