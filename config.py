CONFIG = {
    "max_words": 30,
    "context_size": 10,
    "temperature_default": 0.7,
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
}