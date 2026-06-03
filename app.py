import json
import os
import streamlit as st

from rl import PPOServer
from utils import guardar_log_recompensa
from config import CONFIG
from utils.traducciones import tr


LOGS_FILE = "logs/logs_alineacion.json"
LOGS_FILE_VIEJO = "logs/logs_alineacion.jsonl"


def _migrar_logs_viejo():
    if os.path.exists(LOGS_FILE_VIEJO) and not os.path.exists(LOGS_FILE):
        logs = []
        with open(LOGS_FILE_VIEJO, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        with open(LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        os.remove(LOGS_FILE_VIEJO)


def _leer_logs():
    if not os.path.exists(LOGS_FILE):
        return []
    with open(LOGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def reentrenar_modelo(recompensa):
    generador_rl = st.session_state.get("generador_rl")
    if generador_rl is None:
        return
    trajectory = generador_rl.last_trajectory
    if trajectory and trajectory["actions"]:
        generador_rl.entrenar(trajectory, recompensa)


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
if "idioma" not in st.session_state:
    st.session_state.idioma = "es"
if "generador_rl" not in st.session_state:
    st.session_state.generador_rl = PPOServer()
    st.session_state.generador_rl.inicializar()

_migrar_logs_viejo()

idioma = st.session_state.idioma

st.title(tr("titulo", idioma))

with st.sidebar:
    st.header(tr("config", idioma))

    nuevo_idioma = st.selectbox(
        tr("idioma", idioma),
        options=["es", "en"],
        index=0 if idioma == "es" else 1,
    )
    if nuevo_idioma != idioma:
        st.session_state.idioma = nuevo_idioma
        generos_nuevos = tr("generos", nuevo_idioma)
        tonos_nuevos = tr("tonos", nuevo_idioma)
        st.session_state.genero = generos_nuevos[0]
        st.session_state.tono = tonos_nuevos[0]
        st.rerun()

    consigna = st.text_area(
        tr("consigna", idioma),
        value=st.session_state.consigna,
        height=100,
        placeholder=tr("placeholder_consigna", idioma),
    )

    generos_opts = tr("generos", idioma)
    tonos_opts = tr("tonos", idioma)

    genero = st.selectbox(
        tr("genero", idioma),
        options=generos_opts,
        index=generos_opts.index(st.session_state.genero) if st.session_state.genero in generos_opts else 0,
    )

    tono = st.selectbox(
        tr("tono", idioma),
        options=tonos_opts,
        index=tonos_opts.index(st.session_state.tono) if st.session_state.tono in tonos_opts else 0,
    )

    temperatura = st.slider(
        tr("temperatura", idioma),
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperatura,
        step=0.1,
    )

    st.caption(tr("caption_ppo", idioma))

col1, col2 = st.columns([3, 1])

with col1:
    if st.button(tr("generar", idioma), type="primary", use_container_width=True):
        generador_rl = st.session_state.generador_rl
        texto = generador_rl.generar_con_temperatura(
            num_palabras=CONFIG["max_words"],
            temperatura=temperatura,
            consigna=consigna,
            genero=genero,
            tono=tono,
            idioma=idioma,
        )

        st.session_state.texto_generado = texto
        st.session_state.consigna = consigna
        st.session_state.genero = genero
        st.session_state.tono = tono
        st.session_state.temperatura = temperatura

        st.rerun()

with col2:
    if st.button(tr("limpiar", idioma), use_container_width=True):
        st.session_state.texto_generado = None
        st.rerun()

if st.session_state.texto_generado:
    st.markdown(f"### {tr('texto_generado', idioma)}")
    st.info(st.session_state.texto_generado)

    st.markdown(f"### {tr('feedback', idioma)}")

    col_pos, col_neg = st.columns(2)

    with col_pos:
        if st.button(tr("me_gusta", idioma), use_container_width=True):
            guardar_log_recompensa(
                st.session_state.consigna,
                st.session_state.genero,
                st.session_state.tono,
                st.session_state.temperatura,
                st.session_state.texto_generado,
                1.0,
            )
            reentrenar_modelo(1.0)
            st.success(tr("gracias_feedback", idioma))
            st.balloons()

    with col_neg:
        if st.button(tr("no_gusta", idioma), use_container_width=True):
            guardar_log_recompensa(
                st.session_state.consigna,
                st.session_state.genero,
                st.session_state.tono,
                st.session_state.temperatura,
                st.session_state.texto_generado,
                -1.0,
            )
            reentrenar_modelo(-1.0)
            st.warning(tr("feedback_negativo", idioma))

    st.markdown("---")

    with st.expander(tr("ver_logs", idioma)):
        logs = _leer_logs()
        if logs:
            st.write(f"{tr('total_entradas', idioma)}: {len(logs)}")
            st.json(logs[-5:] if len(logs) > 5 else logs)
        else:
            st.info(tr("sin_logs", idioma))
