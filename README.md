# Escritura Creativa con PPO + Lógica de Primer Orden

Generador de textos creativos en español usando **PPO (Proximal Policy Optimization)** con recompensa basada en **Lógica de Primer Orden** y modelo de **bigramas** estadístico.

## 🔧 Librerías utilizadas

| Librería | Versión | Propósito |
|----------|---------|-----------|
| `streamlit` | 1.57.0 | Interfaz web interactiva |
| `gymnasium` | 1.2.3 | Entorno de Reinforcement Learning |
| `torch` | 2.12.0 | Red neuronal (MLP) del policy PPO |
| `stable-baselines3` | 2.8.0 | Implementación del algoritmo PPO |
| `numpy` | 2.4.6 | Cálculos numéricos y matrices |
| `spaCy` | *indirecta* | Análisis lingüístico del corpus (solo para construir vocabulario) |
| `es_core_news_sm` | *indirecta* | Modelo de lenguaje español de spaCy |

## ⚙️ Funcionamiento

### 1. Vocabulario (195 palabras categorizadas)

Cada palabra del vocabulario pertenece a una categoría sintáctica: **determinante**, **sustantivo**, **verbo**, **preposición**, **conjunción**, **pronombre**, **adjetivo** o **adverbio**. Todos los verbos están en forma conjugada (sin infinitivos).

### 2. Templates ocultos

Hay 14 plantillas sintácticas (ej: `DET SUST VERBO PREP DET SUST`). El modelo elige una al azar y rellena cada slot con palabras de la categoría correspondiente.

### 3. Red PPO (MLP)

La red recibe un vector de observación de 43 dimensiones:
- 3 one-hot para el género (Cuento/Poema/Guion)
- 3 one-hot para el tono (Misterioso/Humorístico/Dramático)
- 20 tokens del prompt
- 15 tokens de contexto
- 2 extras (categoría anterior + posición)

La red aprende qué palabras asignar mayor probabilidad en cada paso.

### 4. Modelo Bigrama

Matriz 197×197 entrenada sobre 5 obras literarias españolas (~15.800 oraciones). En cada paso, la probabilidad final mezcla PPO (65%) y bigrama (35%):

```
P_final = 0.65 × P_PPO + 0.35 × P_bigram
```

Esto hace que las transiciones entre palabras suenen naturales ("era → el/un/la", "de → la/los/las").

### 5. Recompensa (Lógica de Primer Orden)

En cada paso se evalúa:
- **Semántica**: ¿la palabra pertenece al género/tono? ¿es relevante al prompt? ¿es conjugada?
- **Estructura**: reglas sintácticas (ej: determinante → sustantivo, preposición → sustantivo/determinante)

El usuario también puede dar feedback directo (👍/👎) que se usa para reentrenar en línea.

### 6. Post-procesamiento

- Concordancia DET–SUST en género
- Capitalización de la primera letra
- Punto al final

## 🚀 Cómo ejecutar

```bash
.venv\Scripts\activate
streamlit run app.py
```

## 🧠 Cómo mejorar el modelo con feedback

1. Genera un texto con cualquier combinación
2. Dale **👍** si te gusta o **👎** si no
3. El modelo se reentrena al instante con tu feedback
4. Tras 15-20 feedbacks el modelo se ajusta a tus preferencias

## 📁 Estructura del proyecto

```
escritura_creativa/
├── app.py                  # Interfaz Streamlit
├── config.py               # Configuración central
├── requirements.txt        # Dependencias
├── core/
│   ├── vocabulary.py       # Carga y consulta del vocabulario
│   ├── vocab.json          # Vocabulario (195 palabras)
│   ├── logica.py           # Evaluador LPO (recompensa)
│   ├── bigram.py           # Modelo de bigramas
│   ├── bigram_matrix.npy   # Matriz bigrama cacheada
│   ├── posproceso.py       # Post-procesamiento (concordancia, puntuación)
│   ├── evaluator.py        # Prompt para evaluar con IA externa
│   └── dataset.py          # Dataset sintético para pretrain
├── env/
│   └── rl_env.py           # Entorno Gymnasium
├── rl/
│   └── agent.py            # Agente PPO + generación
├── utils/
│   ├── feedback.py         # Logging de feedback
│   └── traducciones.py     # i18n español/inglés
└── models/
    └── modelo_ppo_es.pth   # Pesos del modelo entrenado
```
