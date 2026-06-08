import os
import json

_FALLBACK_WORDS = [
    "el", "la", "un", "una", "los", "las", "este", "esta",
    "su", "mi", "tu", "cada", "todo", "otro", "mismo",
    "de", "en", "con", "por", "para", "sin", "del", "al",
    "entre", "sobre", "bajo",
    "y", "o", "pero", "que", "cuando", "mientras", "porque", "si", "como",
    "se", "le", "lo", "la", "ella", "ellos", "esto", "quien",
    "es", "está", "tiene", "hace", "puede", "dice", "era", "había",
    "va", "mira", "siente", "sueña", "vive", "muere", "ama",
    "sabe", "quiere", "ve", "habla", "oye", "ríe", "llora",
    "camina", "corre", "canta", "baila", "juega",
    "vida", "muerte", "tiempo", "noche", "día", "casa", "camino",
    "bosque", "mar", "fuego", "cielo", "luz", "sombra",
    "amor", "corazón", "alma", "silencio", "voz", "sueño", "secreto",
    "rey", "reina", "mundo", "guerra", "paz", "verdad",
    "agua", "tierra", "luna", "sol", "flor", "estrella",
    "escena", "actor", "dragón", "miedo", "terror",
    "alegría", "dolor", "lágrima", "destino", "honor",
    "grande", "pequeño", "bueno", "malo", "hermoso", "oscuro", "claro",
    "nuevo", "viejo", "fuerte", "feliz", "triste",
    "aquí", "allí", "ahora", "antes", "después", "siempre", "nunca",
    "más", "ya", "entonces", "así",
]

_FALLBACK_CATS = {
    "el": "determinante", "la": "determinante", "un": "determinante", "una": "determinante",
    "los": "determinante", "las": "determinante", "este": "determinante", "esta": "determinante",
    "su": "determinante", "mi": "determinante", "tu": "determinante", "cada": "determinante",
    "todo": "determinante", "otro": "determinante", "mismo": "determinante",
    "de": "preposicion", "en": "preposicion", "con": "preposicion", "por": "preposicion",
    "para": "preposicion", "sin": "preposicion", "del": "preposicion", "al": "preposicion",
    "entre": "preposicion", "sobre": "preposicion", "bajo": "preposicion",
    "y": "conjuncion", "o": "conjuncion", "pero": "conjuncion", "que": "conjuncion",
    "cuando": "conjuncion", "mientras": "conjuncion", "porque": "conjuncion", "si": "conjuncion",
    "como": "conjuncion",
    "se": "pronombre", "le": "pronombre", "lo": "pronombre", "ella": "pronombre",
    "ellos": "pronombre", "esto": "pronombre", "quien": "pronombre",
    "es": "verbo", "está": "verbo", "tiene": "verbo", "hace": "verbo", "puede": "verbo",
    "dice": "verbo", "era": "verbo", "había": "verbo", "va": "verbo", "mira": "verbo",
    "siente": "verbo", "sueña": "verbo", "vive": "verbo", "muere": "verbo", "ama": "verbo",
    "sabe": "verbo", "quiere": "verbo", "ve": "verbo", "habla": "verbo", "oye": "verbo",
    "ríe": "verbo", "llora": "verbo", "camina": "verbo", "corre": "verbo",
    "canta": "verbo", "baila": "verbo", "juega": "verbo",
    "grande": "adjetivo", "pequeño": "adjetivo", "bueno": "adjetivo", "malo": "adjetivo",
    "hermoso": "adjetivo", "oscuro": "adjetivo", "claro": "adjetivo",
    "nuevo": "adjetivo", "viejo": "adjetivo", "fuerte": "adjetivo", "feliz": "adjetivo",
    "triste": "adjetivo",
    "aquí": "adverbio", "allí": "adverbio", "ahora": "adverbio", "antes": "adverbio",
    "después": "adverbio", "siempre": "adverbio", "nunca": "adverbio",
    "más": "adverbio", "ya": "adverbio", "entonces": "adverbio", "así": "adverbio",
}


class Vocabulario:
    def __init__(self, idioma="es"):
        self.idioma = idioma
        vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.palabras = data["palabras"]
            self._categorias = data["categorias"]
            self.ind2word = {int(k): v for k, v in data["ind2word"].items()}
            self.word2ind = data["word2ind"]
            self.pad_token = data["pad_token"]
            self.eos_token = data["eos_token"]
            self.vocab_size = data["vocab_size"]
        else:
            self._build_fallback()

    def _build_fallback(self):
        self.palabras = list(_FALLBACK_WORDS)
        self.palabras.sort()
        self.ind2word = {i: w for i, w in enumerate(self.palabras)}
        self.word2ind = {w: i for i, w in enumerate(self.palabras)}
        self._categorias = dict(_FALLBACK_CATS)
        self.pad_token = len(self.palabras)
        self.eos_token = len(self.palabras) + 1
        self.vocab_size = len(self.palabras) + 2

    def categoria(self, token):
        palabra = self.ind2word.get(token, "")
        return self._categorias.get(palabra, "sustantivo")
