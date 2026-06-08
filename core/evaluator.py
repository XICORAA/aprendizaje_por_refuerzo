_JSON_TPL = """{{
  "puntuacion_total": <0-10>,
  "coherencia_gramatical": <0-1>,
  "adherencia_consigna": <0-1>,
  "coherencia_tematica": <0-1>,
  "genero": <0-1>,
  "tono": <0-1>,
  "naturalidad": <0-1>,
  "diversidad_lexica": <0-1>,
  "feedback": "<1-3 frases en español señalando lo mejor y lo peor>"
}}"""

PROMPT_EVALUADOR = """Eres un crítico literario experto en español. Evalúa el siguiente texto generado por IA según estos criterios, y responde ÚNICAMENTE con un JSON sin explicaciones.

## Criterios

1. **coherencia_gramatical** (0-1): ¿Las palabras siguen un orden sintáctico correcto del español? ¿Hay concordancia DET-SUST, tiempos verbales coherentes?

2. **adherencia_consigna** (0-1): ¿El texto refleja claramente el tema/palabras clave de la consigna?

3. **coherencia_tematica** (0-1): ¿Las palabras se relacionan entre sí formando una imagen o idea unificada? (evitar transiciones incoherentes como "el silencio baila")

4. **genero** (0-1): ¿El tono y vocabulario son apropiados para el género indicado?

5. **tono** (0-1): ¿La selección léxica transmite correctamente el tono?

6. **naturalidad** (0-1): ¿Suena a español literario natural, no a palabras elegidas al azar?

7. **diversidad_lexica** (0-1): ¿Evita repeticiones excesivas de palabras?

## Salida (JSON ÚNICAMENTE)

```json
""" + _JSON_TPL + """

## Texto a evaluar

Consigna: {consigna}
Género: {genero}
Tono: {tono}

Texto: {texto}
"""


def evaluar_con_ia(texto, consigna, genero, tono):
    """Devuelve el prompt listo para copiar/pegar en ChatGPT/Claude."""
    return PROMPT_EVALUADOR.format(
        texto=texto,
        consigna=consigna,
        genero=genero,
        tono=tono,
    )
