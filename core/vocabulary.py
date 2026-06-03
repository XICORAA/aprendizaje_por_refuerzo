class Vocabulario:
    def __init__(self, idioma="es"):
        self.idioma = idioma
        if idioma == "es":
            self.palabras = self._palabras_es()
            self._det = self._det_es()
            self._prep = self._prep_es()
            self._conj = self._conj_es()
            self._pron = self._pron_es()
            self._verb = self._verb_es()
            self._adj = self._adj_es()
            self._adv = self._adv_es()
        else:
            self.palabras = self._palabras_en()
            self._det = self._det_en()
            self._prep = self._prep_en()
            self._conj = self._conj_en()
            self._pron = self._pron_en()
            self._verb = self._verb_en()
            self._adj = self._adj_en()
            self._adv = self._adv_en()
        self.palabras = sorted(set(self.palabras))
        self.ind2word = {i: w for i, w in enumerate(self.palabras)}
        self.word2ind = {w: i for i, w in enumerate(self.palabras)}
        self.pad_token = len(self.palabras)
        self.eos_token = len(self.palabras) + 1
        self.vocab_size = len(self.palabras) + 2
        self._categorias = self._build_categorias()

    def _palabras_es(self):
        return [
            "el", "la", "un", "una", "los", "las", "este", "esta",
            "su", "mi", "tu", "cada", "todo", "otro", "mismo",
            "de", "en", "con", "por", "para", "sin", "del", "al",
            "entre", "sobre", "bajo",
            "y", "o", "pero", "que", "cuando", "mientras", "porque", "si", "como",
            "se", "le", "lo", "ella", "ellos", "esto", "quien",
            "ser", "estar", "tener", "hacer", "poder", "decir", "era", "había",
            "ir", "mirar", "sentir", "soñar", "vivir", "morir", "amar",
            "saber", "querer", "ver", "hablar", "oír", "reír", "llorar",
            "caminar", "correr", "cantar", "bailar", "jugar",
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

    def _palabras_en(self):
        return [
            "the", "a", "an", "this", "that", "my", "your", "his", "her",
            "each", "every", "all", "some", "other", "same",
            "of", "in", "to", "with", "by", "for", "without", "from",
            "between", "about", "under",
            "and", "or", "but", "that", "when", "while", "because", "if", "as",
            "he", "she", "it", "they", "we", "who", "this",
            "be", "have", "do", "can", "say", "see", "was", "were", "had",
            "go", "know", "want", "look", "feel", "dream",
            "live", "die", "love", "fear", "speak", "hear",
            "laugh", "cry", "walk", "run", "sing", "dance", "play",
            "life", "death", "time", "night", "day", "house", "path",
            "forest", "sea", "fire", "sky", "light", "shadow",
            "love", "heart", "soul", "silence", "voice", "dream", "secret",
            "king", "queen", "world", "war", "peace", "truth",
            "water", "earth", "moon", "sun", "flower", "star",
            "scene", "actor", "dragon", "fear", "terror",
            "joy", "pain", "tear", "fate", "honor",
            "big", "small", "good", "bad", "beautiful", "dark", "light",
            "new", "old", "strong", "happy", "sad",
            "here", "there", "now", "before", "after", "always", "never",
            "more", "yet", "then", "so",
        ]

    def _det_es(self):
        return {"el", "la", "un", "una", "los", "las", "este", "esta",
                "su", "mi", "tu", "cada", "todo", "otro", "mismo"}

    def _prep_es(self):
        return {"de", "en", "con", "por", "para", "sin", "del", "al",
                "entre", "sobre", "bajo"}

    def _conj_es(self):
        return {"y", "o", "pero", "que", "cuando", "mientras", "porque", "si", "como"}

    def _pron_es(self):
        return {"se", "le", "lo", "ella", "ellos", "esto", "quien"}

    def _verb_es(self):
        return {"ser", "estar", "tener", "hacer", "poder", "decir", "era", "había",
                "ir", "mirar", "sentir", "soñar", "vivir", "morir", "amar",
                "saber", "querer", "ver", "hablar", "oír", "reír", "llorar",
                "caminar", "correr", "cantar", "bailar", "jugar"}

    def _adj_es(self):
        return {"grande", "pequeño", "bueno", "malo", "hermoso", "oscuro", "claro",
                "nuevo", "viejo", "fuerte", "feliz", "triste"}

    def _adv_es(self):
        return {"aquí", "allí", "ahora", "antes", "después", "siempre", "nunca",
                "más", "ya", "entonces", "así"}

    def _det_en(self):
        return {"the", "a", "an", "this", "that", "my", "your", "his", "her",
                "each", "every", "all", "some", "other", "same"}

    def _prep_en(self):
        return {"of", "in", "to", "with", "by", "for", "without", "from",
                "between", "about", "under"}

    def _conj_en(self):
        return {"and", "or", "but", "that", "when", "while", "because", "if", "as"}

    def _pron_en(self):
        return {"he", "she", "it", "they", "we", "who", "this"}

    def _verb_en(self):
        return {"be", "have", "do", "can", "say", "see", "was", "were", "had",
                "go", "know", "want", "look", "feel", "dream",
                "live", "die", "love", "fear", "speak", "hear",
                "laugh", "cry", "walk", "run", "sing", "dance", "play"}

    def _adj_en(self):
        return {"big", "small", "good", "bad", "beautiful", "dark", "light",
                "new", "old", "strong", "happy", "sad"}

    def _adv_en(self):
        return {"here", "there", "now", "before", "after", "always", "never",
                "more", "yet", "then", "so"}

    def _build_categorias(self):
        cats = {}
        for w in self.palabras:
            if w in self._det:
                cats[w] = "determinante" if self.idioma == "es" else "determiner"
            elif w in self._prep:
                cats[w] = "preposicion" if self.idioma == "es" else "preposition"
            elif w in self._conj:
                cats[w] = "conjuncion" if self.idioma == "es" else "conjunction"
            elif w in self._pron:
                cats[w] = "pronombre" if self.idioma == "es" else "pronoun"
            elif w in self._verb:
                cats[w] = "verbo" if self.idioma == "es" else "verb"
            elif w in self._adj:
                cats[w] = "adjetivo" if self.idioma == "es" else "adjective"
            elif w in self._adv:
                cats[w] = "adverbio" if self.idioma == "es" else "adverb"
            else:
                cats[w] = "sustantivo" if self.idioma == "es" else "noun"
        return cats

    def categoria(self, token):
        palabra = self.ind2word.get(token, "")
        return self._categorias.get(palabra, "sustantivo" if self.idioma == "es" else "noun")
