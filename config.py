CONFIG = {
    "max_words": 15,
    "context_size": 15,
    "temperature_default": 0.7,
    "prompt_max_tokens": 20,
    "lambda_semantica": 0.5,
    "lambda_estructura": 1.0,
    "logica_como_recompensa": True,
    "ppo_config": {
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 4,
        "gamma": 0.99,
    },
    "generos": ["Cuento", "Poema", "Guion"],
    "tonos": ["Misterioso", "Humorístico", "Dramático"],
    "genero_prefixes": {
        "Cuento": ["había", "en", "un", "el", "la"],
        "Poema": ["en", "con", "entre", "sobre", "bajo"],
        "Guion": ["entonces", "después", "mientras", "cuando", "así"],
    },
    "palabras_por_genero": {
        "Cuento": [
            "era", "había", "una", "cuando", "casa", "camino", "bosque",
            "rey", "reina", "vida", "muerte", "sueño", "secreto",
            "mundo", "guerra", "paz", "amor", "corazón", "alma",
            "silencio", "dragón", "noche", "luz", "sombra",
        ],
        "Poema": [
            "luna", "sol", "estrella", "flor", "cielo", "mar", "amor",
            "corazón", "alma", "silencio", "vida", "muerte", "sueño",
            "luz", "sombra", "noche", "día", "agua", "fuego", "tierra", "voz",
        ],
        "Guion": [
            "entonces", "después", "cuando", "mientras", "escena",
            "actor", "dice", "mira", "siente", "habla",
            "voz", "silencio",
        ],
    },
    "palabras_por_tono": {
        "Misterioso": [
            "noche", "sombra", "oscuro", "secreto", "silencio",
            "muerte", "bosque", "luna", "miedo", "terror",
            "sueño", "alma", "corazón", "voz",
        ],
        "Humorístico": [
            "ríe", "alegría", "feliz", "amigo", "juega",
            "canta", "baila", "sueña", "amor", "bueno",
            "hermoso", "fuerte", "grande",
        ],
        "Dramático": [
            "dolor", "amor", "destino", "corazón", "alma",
            "vida", "muerte", "guerra", "llora", "triste",
            "lucha", "victoria", "derrota", "honor",
            "silencio", "lágrima", "miedo", "terror",
        ],
    },
    "generos_en": ["Tale", "Poem", "Script"],
    "tonos_en": ["Mysterious", "Humorous", "Dramatic"],
    "palabras_por_genero_en": {
        "Tale": [
            "once", "upon", "time", "house", "path", "forest",
            "king", "queen", "life", "death", "dream", "secret",
            "world", "war", "peace", "love", "heart", "soul",
            "silence", "dragon", "night", "light", "shadow",
        ],
        "Poem": [
            "moon", "sun", "star", "flower", "sky", "sea", "love",
            "heart", "soul", "silence", "life", "death", "dream",
            "light", "shadow", "night", "day", "water", "fire", "earth", "voice",
        ],
        "Script": [
            "then", "after", "when", "while", "scene",
            "actor", "says", "looks", "feels", "speaks",
            "voice", "silence",
        ],
    },
    "palabras_por_tono_en": {
        "Mysterious": [
            "night", "shadow", "dark", "secret", "silence",
            "death", "forest", "moon", "fear", "terror",
            "dream", "soul", "heart", "voice",
        ],
        "Humorous": [
            "laughs", "joy", "happy", "friend", "plays",
            "sings", "dances", "dreams", "love", "good",
            "beautiful", "strong", "big",
        ],
        "Dramatic": [
            "pain", "love", "fate", "heart", "soul",
            "life", "death", "war", "tears", "sad",
            "struggle", "victory", "defeat", "honor",
            "silence", "fear", "terror",
        ],
    },
}
