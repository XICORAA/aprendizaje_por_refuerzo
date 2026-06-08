_GENERO = {
    "vida": "f", "muerte": "f", "noche": "f", "casa": "f", "luz": "f",
    "sombra": "f", "alma": "f", "voz": "f", "reina": "f", "paz": "f",
    "verdad": "f", "agua": "f", "tierra": "f", "luna": "f", "flor": "f",
    "estrella": "f", "escena": "f", "alegría": "f", "lágrima": "f",
    "tiempo": "m", "día": "m", "camino": "m", "bosque": "m", "mar": "m",
    "fuego": "m", "cielo": "m", "amor": "m", "corazón": "m",
    "silencio": "m", "sueño": "m", "secreto": "m", "rey": "m",
    "mundo": "m", "sol": "m", "actor": "m", "dragón": "m", "miedo": "m",
    "terror": "m", "dolor": "m", "destino": "m", "honor": "m",
}

_DET_MASC = {"el", "un", "los", "este", "todo", "otro", "mismo"}
_DET_FEM = {"la", "una", "las", "esta"}
_DET_INV = {"su", "mi", "tu", "cada"}

def concordar_frase(texto, vocabulario):
    palabras = texto.split()
    if not palabras:
        return texto
    for i in range(len(palabras) - 1):
        if vocabulario._categorias.get(palabras[i], "") == "determinante":
            det = palabras[i]
            sig = palabras[i + 1]
            gen = _GENERO.get(sig, "")
            if gen == "m" and det in _DET_FEM:
                idx = list(_DET_FEM).index(det)
                palabras[i] = list(_DET_MASC)[idx] if idx < len(_DET_MASC) else det
            elif gen == "f" and det in _DET_MASC:
                idx = list(_DET_MASC).index(det)
                palabras[i] = list(_DET_FEM)[idx] if idx < len(_DET_FEM) else det
    palabras[0] = palabras[0].capitalize()
    return " ".join(palabras) + "."
