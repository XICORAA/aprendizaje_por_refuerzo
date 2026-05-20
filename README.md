# Escritura Creativa con RL

Proyecto de escritura creativa en español usando aprendizaje por refuerzo.

## Requisitos

- Python 3.10+

## Instalacion

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual (Windows)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecucion

```bash
streamlit run app.py
```

La aplicacion se abrira en http://localhost:8501

## Uso

1. Selecciona el modo: **RL (PPO)** o **Bigrama**
2. Ingresa una consigna (opcional)
3. Elige genero (Cuento, Poema, Guion) y tono (Misterioso, Humoristico, Dramatico)
4. Ajusta la temperatura (0 = determinista, 1 = creativo)
5. Click en "Generar Texto"
6. Proporciona feedback thumbs up/down para mejorar el modelo

## Estructura

```
app.py          - Interfaz Streamlit
config.py       - Configuracion
core/           - Generador bigrama y vocabulario
env/            - Entorno Gymnasium RL
rl/             - Agente PPO
utils/          - Feedback y logging
```