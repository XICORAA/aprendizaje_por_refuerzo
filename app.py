import streamlit as st
import os

from core import GeneradorBigrama
from rl import PPOServer
from utils import guardar_log_recompensa
from config import CONFIG


st.set_page_config(page_title="Escritura Creativa con RL")

if "texto_generado" not in st.session_state:
    st.session_state.texto_generado = None
if "consigna" not in st.session_state:
    st.session_state.consigna = ""
if "genero" not in st.session_state:
    st.session_state.genero = "Cuento"
if "tono" not in st.session_state:
    st.session_state.tono = "Misterioso"
if "temperatura" not in st.session_state:
    st.session_state.temperatura = CONFIG["temperature_default"]
if "modo_rl" not in st.session_state:
    st.session_state.modo_rl = False

st.title("Asistente de Escritura Creativa con RL")
st.markdown("---")

with st.sidebar:
    st.header("Configuracion")

    st.session_state.modo_rl = st.toggle(
        "Usar RL (PPO)",
        value=st.session_state.modo_rl,
        help="Activa el modelo PPO entrenado. Si está desactivado, usa bigrama simple."
    )

    st.caption("Modo: " + ("RL" if st.session_state.modo_rl else "Bigrama"))

    consigna = st.text_area(
        "Consigna",
        value=st.session_state.consigna,
        height=100,
        placeholder="Ej: Escribe una historia sobre un detective..."
    )

    genero = st.selectbox(
        "Género",
        options=CONFIG["generos"],
        index=CONFIG["generos"].index(st.session_state.genero)
    )

    tono = st.selectbox(
        "Tono",
        options=CONFIG["tonos"],
        index=CONFIG["tonos"].index(st.session_state.tono)
    )

    temperatura = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperatura,
        step=0.1,
        help="0.0 = más determinista, 1.0 = más creativo"
    )

    st.markdown("---")
    st.caption("La temperatura controla la aleatoriedad en la generacion de texto.")

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("Generar Texto", type="primary", use_container_width=True):
        if st.session_state.modo_rl:
            generador_rl = PPOServer()
            texto = generador_rl.generar_con_temperatura(
                num_palabras=CONFIG["max_words"],
                temperatura=temperatura
            )
        else:
            generador_bg = GeneradorBigrama()
            texto = generador_bg.generar(
                num_palabras=CONFIG["max_words"],
                temperatura=temperatura,
                genero=genero,
                tono=tono
            )

        st.session_state.texto_generado = texto
        st.session_state.consigna = consigna
        st.session_state.genero = genero
        st.session_state.tono = tono
        st.session_state.temperatura = temperatura

        st.rerun()

with col2:
    if st.button("Limpiar", use_container_width=True):
        st.session_state.texto_generado = None
        st.rerun()

if st.session_state.texto_generado:
    st.markdown("### Texto Generado")
    st.info(st.session_state.texto_generado)

    modo_actual = "RL" if st.session_state.modo_rl else "Bigrama"
    st.caption(f"Generado con: {modo_actual}")

    st.markdown("### Feedback")

    col_pos, col_neg = st.columns(2)

    with col_pos:
        if st.button("Coherente / Me gusta", use_container_width=True):
            guardar_log_recompensa(
                st.session_state.consigna,
                st.session_state.genero,
                st.session_state.tono,
                st.session_state.temperatura,
                st.session_state.texto_generado,
                1.0
            )
            st.success("¡Gracias! Feedback positivo guardado.")
            st.balloons()

    with col_neg:
        if st.button("Incoherente / No me gusta", use_container_width=True):
            guardar_log_recompensa(
                st.session_state.consigna,
                st.session_state.genero,
                st.session_state.tono,
                st.session_state.temperatura,
                st.session_state.texto_generado,
                -1.0
            )
            st.warning("Feedback negativo guardado. El modelo aprenderá de esto.")

    st.markdown("---")

    with st.expander("Ver logs de alineacion"):
        if os.path.exists("logs_alineacion.json"):
            with open("logs_alineacion.json", "r", encoding="utf-8") as f:
                logs = f.read()
            if logs.strip():
                import json
                logs = json.loads(logs)
                st.write(f"Total de entradas: {len(logs)}")
                st.json(logs[-5:] if len(logs) > 5 else logs)
            else:
                st.info("No hay logs aún.")
        else:
            st.info("No hay logs aún. Genera texto y da feedback.")


def reentrenar_modelo():
    pass