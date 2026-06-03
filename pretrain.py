import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from rl.agent import PPOServer, _modelo_ruta
from core.dataset import DatasetOraciones
from config import CONFIG


def pretrain():
    server = PPOServer(idioma="es")
    server.inicializar()
    print(f"  Modelo cargado (vocab_size={server.vocabulario.vocab_size})")

    dataset = DatasetOraciones(server.vocabulario)
    print(f"  {len(dataset)} oraciones, ~{sum(len(e['tokens'])-1 for e in dataset.ejemplos)} posiciones de entrenamiento")

    lote = []
    batch_size = 32
    total_pasos = 0
    policy = server.agent.policy
    optim = policy.optimizer
    ce_loss = torch.nn.CrossEntropyLoss()

    NUM_EPOCAS = 200
    log_interval = 20
    mejor_perdida = float("inf")

    for epoca in range(1, NUM_EPOCAS + 1):
        perdidas = []
        policy.train()

        for obs, target in dataset.generar_lotes(batch_size=batch_size):
            lote.append((obs, target))

            if len(lote) >= batch_size:
                obs_batch = torch.tensor(
                    np.array([x[0] for x in lote]), dtype=torch.float32
                )
                target_batch = torch.tensor(
                    [x[1] for x in lote], dtype=torch.long
                )

                features = policy.extract_features(obs_batch)
                latent_pi, _ = policy.mlp_extractor(features)
                logits = policy.action_net(latent_pi)

                loss = ce_loss(logits, target_batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                perdidas.append(float(loss))
                total_pasos += 1
                lote = []

        if lote:
            obs_batch = torch.tensor(np.array([x[0] for x in lote]), dtype=torch.float32)
            target_batch = torch.tensor([x[1] for x in lote], dtype=torch.long)
            features = policy.extract_features(obs_batch)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
            loss = ce_loss(logits, target_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            perdidas.append(float(loss))
            lote = []

        perdida_media = np.mean(perdidas) if perdidas else 0.0

        if perdida_media < mejor_perdida:
            mejor_perdida = perdida_media
            server.guardar()

        if epoca == 1 or epoca % log_interval == 0 or epoca == NUM_EPOCAS:
            print(f"  E{epoca:3d}/{NUM_EPOCAS}  loss={perdida_media:.4f}  pasos={total_pasos}  mejor={mejor_perdida:.4f}")

    print(f"\n  PRETRAIN COMPLETO ({NUM_EPOCAS} epocas, {total_pasos} pasos)")
    print(f"  Mejor loss: {mejor_perdida:.4f}")
    print(f"  Modelo guardado en {_modelo_ruta('es')}")


if __name__ == "__main__":
    t0 = time.time()
    pretrain()
    t1 = time.time()
    print(f"  Tiempo: {t1 - t0:.0f}s")
