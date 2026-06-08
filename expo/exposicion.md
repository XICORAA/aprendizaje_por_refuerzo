# Exposición: Cerebro del generador — PPO, Bigrama y Lógica de Primer Orden

## 1. Arquitectura general del "cerebro"

El generador no es una red neuronal gigante tipo GPT. Es un sistema híbrido con tres componentes que trabajan juntos:

```
Observación (43 dims)
       ↓
┌─────────────────────────────┐
│ ① Red PPO (MLP Extractor)   │  ←  rl/agent.py
│    → features latentes       │
│    → distribución de prob.   │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│ ② Modelo Bigrama            │  ←  core/bigram.py
│    → matriz 197×197          │
│    → transiciones del corpus │
└──────────┬──────────────────┘
           ↓
     P_final = 0.65·PPO + 0.35·Bigram
           ↓
┌─────────────────────────────┐
│ ③ Lógica de Primer Orden    │  ←  core/logica.py
│    → recompensa semántica    │
│    → recompensa estructural  │
│    → guía el entrenamiento   │
└─────────────────────────────┘
```

---

## 2. Componente ① — Red PPO (el "cerebro neuronal")

### Archivo: `rl/agent.py` — clase `PPOServer`

#### 2.1 Observación (input)

La red recibe un vector de **43 dimensiones** (`rl/agent.py:159-168`):

```
[ género(3) | tono(3) | prompt(20) | contexto(15) | cat_anterior(1) | posición(1) ]
```

| Bloque | Dims | Qué codifica |
|--------|------|-------------|
| género | 3 | one-hot: Cuento / Poema / Guion |
| tono | 3 | one-hot: Misterioso / Humorístico / Dramático |
| prompt | 20 | IDs de palabras de la consigna (padding si <20) |
| contexto | 15 | IDs de las últimas palabras generadas |
| cat_anterior | 1 | índice numérico de la categoría sintáctica anterior |
| posición | 1 | paso actual en la generación |

#### 2.2 MLP Extractor (el "córtex")

Arquitectura definida por `stable-baselines3` + `ActorCriticPolicy`:

```
Input (43)
  → SharedNet [64 → 64 → 64]  (ReLU)
    → PiNetwork [64 → 64]      (policy head)
      → ActionDistribution (197 logits)
    → VfNetwork [64 → 64]      (value head)
      → Valor escalar V(s)
```

- **Policy head**: produce logits → softmax → P(a|s) para cada una de las 197 palabras.
- **Value head**: estima el valor esperado V(s) del estado actual.

**Código**: `rl/agent.py:76-92` — inicialización del PPO con `ActorCriticPolicy` y `MlpExtractor`.

#### 2.3 Generación (forward)

En `generar_con_temperatura()` (`rl/agent.py:295-360`):

1. `probs = self._get_action_probs(obs, temperatura)` → obtiene P(a|s) de la red
2. Se mezcla con bigrama: `probs = 0.65 * probs + 0.35 * bigram_p`
3. Se aplican sesgos semánticos (género/tono/relevancia)
4. Se enmascaran categorías inválidas según el template
5. Se aplican penalizaciones por repetición y frecuencia global
6. Top-k sampling (k=25)
7. Se muestrea la palabra

#### 2.4 Entrenamiento (backward)

El algoritmo PPO actualiza los pesos usando:

```
L_clip = -E[ min(ratio · A, clip(ratio, 1-ε, 1+ε) · A) ]
ratio = π_nuevo(a|s) / π_viejo(a|s)
A = R - V(s)   # ventaja
```

Donde:
- `R` es la recompensa acumulada (MC returns)
- `V(s)` es el valor estimado por la value head
- ε = 0.2 (clip)

**Código**: `rl/agent.py:195-260` — método `entrenar()` con 3 inner updates por batch.

---

## 3. Componente ② — Modelo Bigrama (el "cerebelo")

### Archivo: `core/bigram.py` — clase `BigramModel`

#### 3.1 Entrenamiento

Se procesan 5 obras literarias españolas (~15.828 oraciones):

```
bigram[palabra_anterior][palabra_siguiente] += 1
```

Se aplica suavizado Laplace (+1 a todo): `matrix[p][n] = 1 + count(p → n)`

Luego se normaliza por filas: `P(w_n | w_{n-1}) = matrix[prev][next] / sum(matrix[prev])`

**Resultado**: matriz `(197, 197)` donde cada fila suma 1.0.

Cacheada en `core/bigram_matrix.npy` para carga instantánea.

#### 3.2 Integración

Durante la generación (`rl/agent.py:306-308`):

```python
bigram_p = self._bigram.probs(ultimo_token)
probs = 0.65 * probs + 0.35 * bigram_p  # mezcla lineal
```

**Efecto**: transiciones naturales como "era → el (13%)", "de → la (28%)", "amor → es (2.4%)".

El bigrama funciona como un **prior estadístico**: cuando la PPO no sabe qué palabra elegir (pesos aleatorios al inicio), el bigrama empuja hacia transiciones reales del español literario.

---

## 4. Componente ③ — Lógica de Primer Orden (el "juez")

### Archivo: `core/logica.py` — clase `LogicaPrimerOrden`

La LPO se usa como **función de recompensa** que guía el aprendizaje del PPO. No es una red — es un sistema de reglas formales.

#### 4.1 Conectivas (implementación Lukasiewicz)

```python
negacion(a)    = 1 - a
conjuncion(a,b) = max(0, a + b - 1)
disyuncion(a,b) = min(1, a + b)
implicacion(a,b)= min(1, 1 - a + b)
```

**Archivo**: `core/logica.py:36-48`

#### 4.2 Evaluación semántica (por palabra)

En cada paso, `evaluar_paso()` (`core/logica.py:113-125`) calcula:

```
score(token) = 0.7 × max(es_del_genero, es_del_tono) + 0.3 × es_relevante
if score == 0 and token es función → score = 0.2
if token es verbo conjugado → score += 0.4 (capado a 1.0)
```

Esto premia palabras que:
- Pertenecen al vocabulario esperado para el género/tono
- Están en la consigna del usuario
- Son verbos conjugados (no infinitivos)

#### 4.3 Evaluación estructural (oración completa)

`evaluar_estructura()` (`core/logica.py:153-188`) aplica 5 reglas formales:

| Regla | Expresión LPO | Penaliza si... |
|-------|---------------|----------------|
| R1 | ∀i DET(i) → ∃j∈[i+1,i+3] SUST(j) | un "el" sin sustantivo cerca |
| R2 | ∀i PREP(i) → ∃j∈[i+1,i+4] (SUST(j) ∨ DET(j)) | "de" sin destino |
| R3 | ∀i ∃j∈[i-2,i+2] VERBO(j) | falta de verbo en la ventana |
| R4 | ∀i ADJ(i) → ∃j∈[i-2,i+2]\{i} SUST(j) | adjetivo sin sustantivo |
| R5 | ∀i PRON(i) → VERBO(i+1) | "se" sin verbo después |

Cada regla se evalúa con implicación de Lukasiewicz: `score = min(1, 1 - antecedente + consecuente)`.

#### 4.4 Loss combinada

En el pretrain (`core/logica.py:193-246`):

```
Loss = λ_sem × Loss_sem + λ_est × Loss_est

Donde:
Loss_sem = -mean(log π(a|s) × score_sem)
Loss_est = -mean(log π(a|s) × score_est)
λ_sem = 0.3, λ_est = 0.2
```

Esto es **aprendizaje por policy gradient**: si la LPO da score alto a una palabra, la PPO aumenta su probabilidad; si da score bajo, la reduce.

---

## 5. Flujo completo paso a paso

```
Usuario escribe consigna + elige género/tono
       ↓
app.py llama a agent.generar(consigna, genero, tono)
       ↓
agent.py:
  1. Elige template sintáctico al azar (14 disponibles)
  2. Para cada posición del template:
     a. Construye observación (43 dims) ← rl_env.py
     b. Obtiene P(a|s) de la red PPO
     c. Mezcla con P_bigram(a | palabra_anterior)
     d. Enmascara categorías que no corresponden
     e. Aplica temperatura y top-k
     f. Muestrea palabra
  3. Concatena palabras → frase
  4. Aplica post-procesamiento ← core/posproceso.py
       • Concordancia DET-SUST (género)
       • Capitalización
       • Punto final
       ↓
Devuelve texto al usuario
       ↓
Usuario da 👍/👎
       ↓
agent.entrenar(trajectory, recompensa):
  - Calcula MC returns (R)
  - Calcula ventajas (A = R - V(s))
  - 3 iteraciones de PPO clip
  - Actualiza pesos de la red
```

---

## 6. Mapa de archivos → conceptos

> ⭐ = **archivo principal del cerebro/algoritmo** — donde ocurre la magia

| Prioridad | Concepto | Archivo | Líneas clave |
|-----------|----------|---------|-------------|
| ⭐⭐⭐ | **Red PPO (MLP) + generación + entrenamiento** | `rl/agent.py` | 76-92 (init PPO), 195-260 (entrenar), 295-360 (generar) |
| ⭐⭐⭐ | **Recompensa LPO semántica y estructural** | `core/logica.py` | 113-125 (evaluar_paso), 153-188 (evaluar_estructura), 36-48 (conectivas) |
| ⭐⭐ | **Bigrama matriz 197×197** | `core/bigram.py` | 67-71 (init), 88-107 (entrenar), 109-117 (probs) |
| ⭐⭐ | **Observación 43-dim (input del cerebro)** | `env/rl_env.py` | 100-154 (construcción obs) |
| ⭐ | Vocabulario 197 palabras | `core/vocabulary.py` | 62-86 (carga), `core/vocab.json` |
| ⭐ | Templates sintácticos | `rl/agent.py` | 47-50 (_SLOTS), 270-278 (selección) |
| ⭐ | Post-procesamiento | `core/posproceso.py` | 20-33 (concordar_frase) |
| — | Feedback usuario | `app.py` | 150-175 (botones), `utils/feedback.py` |
| — | Prompt evaluador IA | `core/evaluator.py` | 1-42 (PROMPT_EVALUADOR) |
| — | Configuración | `config.py` | 1-97 (pesos, max_words, temperaturas) |

---

## 7. Resumen matemático

```
π_θ(a|s) = softmax( MLP(obs) )           ← política parametrizada por θ
V_φ(s)   = MLP(obs) → escalar             ← función de valor

P_final(a|s, prev) = 0.65·π_θ(a|s) + 0.35·P_bigram(a|prev)

R_paso = LPO_sem(a, género, tono, prompt)
R_est  = LPO_est(secuencia completa)

J(θ) = E[ min(ratio·A, clip(ratio, 0.8, 1.2)·A) ]
 donde ratio = π_θ / π_θ_old
       A = Σγ^k·R_paso_k - V_φ(s)
```

---

## 8. Conclusión

Este sistema demuestra que **no se necesita un transformer gigante** para generar texto coherente. Con:

- Un **MLP pequeño** (3 capas × 64 neuronas)
- Una **matriz bigrama** (197×197 = ~39K parámetros)
- Un conjunto de **reglas LPO** (5 reglas sintácticas + evaluación semántica)
- **Feedback humano** como señal de refuerzo

Se obtiene un generador que:
- Corre en CPU en tiempo real
- Produce oraciones gramaticalmente correctas
- Se adapta a las preferencias del usuario en minutos
- Es completamente transparente (cada pieza es inspeccionable)
