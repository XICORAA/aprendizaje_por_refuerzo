import os
import re
import numpy as np
from collections import Counter
from urllib.request import urlopen
from urllib.error import URLError
import ssl

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "aprendizaje_por_refuerzo-main",
                          "aprendizaje_por_refuerzo_alpha-main")

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "bigram_matrix.npy")

CORPUS_FILES = {
    "cuentos_maupassant.txt":
        "https://www.gutenberg.org/ebooks/74908.txt.utf-8",
    "becquer_leyendas.txt":
        "https://www.gutenberg.org/files/53552/53552-0.txt",
    "quevedo_suenos1.txt":
        "https://www.gutenberg.org/ebooks/65999.txt.utf-8",
    "quevedo_buscon.txt":
        "https://www.gutenberg.org/ebooks/32315.txt.utf-8",
    "cuentos_amor.txt":
        "https://www.gutenberg.org/files/55514/55514-0.txt",
}


def _descargar_corpus():
    """Descarga los textos literarios si no existen."""
    os.makedirs(CORPUS_DIR, exist_ok=True)
    ctx = ssl._create_unverified_context()
    for nombre, url in CORPUS_FILES.items():
        ruta = os.path.join(CORPUS_DIR, nombre)
        if os.path.exists(ruta):
            continue
        try:
            print(f"  Descargando {nombre}...")
            with urlopen(url, context=ctx, timeout=30) as resp:
                data = resp.read()
            with open(ruta, "wb") as f:
                f.write(data)
            print(f"    OK ({len(data)} bytes)")
        except (URLError, OSError) as e:
            print(f"    Error descargando {nombre}: {e}")


def _cargar_oraciones():
    _descargar_corpus()
    oraciones = []
    for nombre in CORPUS_FILES:
        ruta = os.path.join(CORPUS_DIR, nombre)
        if not os.path.exists(ruta):
            continue
        try:
            with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read()
        except Exception:
            continue
        inicio = texto.find("*** START")
        if inicio >= 0:
            texto = texto[inicio:]
        fin = texto.find("*** END")
        if fin >= 0:
            texto = texto[:fin]
        texto = re.sub(r'\[.*?\]|\*+|[_\-\[\]"\'<>]', ' ', texto)
        texto = re.sub(r'\d+', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        for s in re.split(r'[.!?]+', texto):
            s = s.strip().lower()
            if len(s.split()) >= 3:
                oraciones.append(s)
    return oraciones


class BigramModel:
    def __init__(self, vocabulario):
        self.vocab = vocabulario

        if os.path.exists(CACHE_FILE):
            self.matrix = np.load(CACHE_FILE)
        else:
            self.matrix = np.ones(
                (vocabulario.vocab_size, vocabulario.vocab_size),
                dtype=np.float32)
            self._train()
            np.save(CACHE_FILE, self.matrix)

    def _train(self):
        oraciones = _cargar_oraciones()
        counts = Counter()
        word2ind = self.vocab.word2ind

        for s in oraciones:
            tokens = s.split()
            prev = None
            for w in tokens:
                idx = word2ind.get(w)
                if idx is None:
                    prev = None
                    continue
                if prev is not None:
                    counts[(prev, idx)] += 1
                prev = idx

        for (prev, nxt), c in counts.items():
            self.matrix[prev, nxt] += c

        row_sums = self.matrix.sum(axis=1, keepdims=True)
        self.matrix /= row_sums

        n_matches = len(counts)
        n_sentences = len(oraciones)
        print(f"  Bigram: {n_matches} pares únicos de {n_sentences} oraciones")

    def probs(self, prev_token):
        if prev_token is None or prev_token >= self.matrix.shape[0]:
            return np.ones(self.vocab.vocab_size,
                           dtype=np.float32) / self.vocab.vocab_size
        return self.matrix[prev_token].copy()
