# Escritura Creativa con PPO + Lógica de Primer Orden

Generador de textos creativos en español usando **PPO (Proximal Policy Optimization)** con recompensa basada en **Lógica de Primer Orden (Łukasiewicz fuzzy logic)**.

## Cómo ejecutar

```bash
.venv\Scripts\activate
streamlit run app.py
```

## Combinaciones para probar

Para cada combinación, copia los 3 valores (Consigna, Género, Tono) en la UI y genera.

### Cuento / Misterioso
| Consigna | Género | Tono |
|----------|--------|------|
| `un viaje en el bosque encantado` | Cuento | Misterioso |
| `el misterio del castillo abandonado` | Cuento | Misterioso |
| `la noche de la luna roja` | Cuento | Misterioso |
| `el secreto del viejo farero` | Cuento | Misterioso |
| `la biblioteca de los susurros` | Cuento | Misterioso |
| `el cuarto prohibido de la mansion` | Cuento | Misterioso |
| `el espejo que robaba almas` | Cuento | Misterioso |
| `la mascara del entierro` | Cuento | Misterioso |
| `el rio de las sombras eternas` | Cuento | Misterioso |
| `el arbol que crecia al reves` | Cuento | Misterioso |

### Cuento / Dramático
| Consigna | Género | Tono |
|----------|--------|------|
| `el principe rescata a la princesa` | Cuento | Dramático |
| `la última batalla del guerrero` | Cuento | Dramático |
| `el amor prohibido del caballero` | Cuento | Dramático |
| `el sacrificio del héroe` | Cuento | Dramático |
| `el regreso del soldado perdido` | Cuento | Dramático |
| `la promesa antes de morir` | Cuento | Dramático |
| `el rey que perdio su reino` | Cuento | Dramático |
| `la hija del traidor` | Cuento | Dramático |
| `el guerrero herido en el campo de batalla` | Cuento | Dramático |
| `el ultimo abrazo del padre` | Cuento | Dramático |

### Cuento / Humorístico
| Consigna | Género | Tono |
|----------|--------|------|
| `el dragon y el heroe torpe` | Cuento | Humorístico |
| `el mago que perdió su sombrero` | Cuento | Humorístico |
| `el caballero y su escudo oxidado` | Cuento | Humorístico |
| `la princesa que quiso ser cantante` | Cuento | Humorístico |
| `el rey que estornudaba sin parar` | Cuento | Humorístico |
| `el principe que le gustaba cocinar` | Cuento | Humorístico |
| `el castillo donde todo salia mal` | Cuento | Humorístico |
| `la bruja que vendia seguros magicos` | Cuento | Humorístico |
| `el dragon vegetariano y la oveja` | Cuento | Humorístico |
| `el mago novato y su varita rota` | Cuento | Humorístico |

### Poema / Misterioso
| Consigna | Género | Tono |
|----------|--------|------|
| `la luna llena de medianoche` | Poema | Misterioso |
| `el susurro del viento en el bosque` | Poema | Misterioso |
| `la niebla que oculta el mar` | Poema | Misterioso |
| `las estrellas en el cielo oscuro` | Poema | Misterioso |
| `la sombra detras de la cortina` | Poema | Misterioso |
| `el eco en el valle vacio` | Poema | Misterioso |
| `el ultimo rayo de sol` | Poema | Misterioso |
| `la voz que llama desde el abismo` | Poema | Misterioso |
| `el arbol seco y la luna` | Poema | Misterioso |
| `el silencio de las ruinas` | Poema | Misterioso |

### Poema / Dramático
| Consigna | Género | Tono |
|----------|--------|------|
| `el amor eterno del poeta` | Poema | Dramático |
| `la soledad del caminante` | Poema | Dramático |
| `el llanto de la noche` | Poema | Dramático |
| `el recuerdo de un adiós` | Poema | Dramático |
| `la carta que nunca envie` | Poema | Dramático |
| `el suspiro del amante olvidado` | Poema | Dramático |
| `la ultima lagrima de la guerra` | Poema | Dramático |
| `el silencio entre dos almas` | Poema | Dramático |
| `el ocaso del corazon herido` | Poema | Dramático |
| `la esperanza en la tormenta` | Poema | Dramático |

### Poema / Humorístico
| Consigna | Género | Tono |
|----------|--------|------|
| `el gato que escribía sonetos` | Poema | Humorístico |
| `la rana que quiso ser princesa` | Poema | Humorístico |
| `el payaso y la luna` | Poema | Humorístico |
| `el perro que bailaba bajo la lluvia` | Poema | Humorístico |
| `el poeta que perdio la rima` | Poema | Humorístico |
| `la musa que no queria inspirar` | Poema | Humorístico |
| `el soneto del cafe frio` | Poema | Humorístico |
| `la oda al calcetin perdido` | Poema | Humorístico |
| `el poema de la tortuga veloz` | Poema | Humorístico |
| `la rima de la vaca voladora` | Poema | Humorístico |

### Guion / Misterioso
| Consigna | Género | Tono |
|----------|--------|------|
| `la ultima escena antes del final` | Guion | Misterioso |
| `el personaje que sabia demasiado` | Guion | Misterioso |
| `el dialogo en la habitacion cerrada` | Guion | Misterioso |
| `la mascara del tercer acto` | Guion | Misterioso |
| `el testigo en la sombra` | Guion | Misterioso |
| `la llamada de media noche` | Guion | Misterioso |
| `el escenario vacio y la voz` | Guion | Misterioso |
| `el actor que desaparecio en escena` | Guion | Misterioso |
| `la carta anonima del camerino` | Guion | Misterioso |
| `el ensayo general del crimen perfecto` | Guion | Misterioso |

### Guion / Dramático
| Consigna | Género | Tono |
|----------|--------|------|
| `la batalla final del acto tres` | Guion | Dramático |
| `el monologo del heroe caido` | Guion | Dramático |
| `el reencuentro despues de la guerra` | Guion | Dramático |
| `el sacrificio del protagonista` | Guion | Dramático |
| `el adios del soldado en el anden` | Guion | Dramático |
| `la confesion antes del juicio` | Guion | Dramático |
| `el duelo entre hermanos` | Guion | Dramático |
| `la ultima palabra del condenado` | Guion | Dramático |
| `el abrazo roto de los amantes` | Guion | Dramático |
| `la decision que cambio todo` | Guion | Dramático |

### Guion / Humorístico
| Consigna | Género | Tono |
|----------|--------|------|
| `el actor que olvido su texto` | Guion | Humorístico |
| `la obra de teatro mas absurda` | Guion | Humorístico |
| `el director y su elenco loco` | Guion | Humorístico |
| `el ensayo que salio mal` | Guion | Humorístico |
| `el guionista sin ideas` | Guion | Humorístico |
| `la escena del pastel en la cara` | Guion | Humorístico |
| `el actor que se quedo dormido en escena` | Guion | Humorístico |
| `la obra donde todos se equivocan` | Guion | Humorístico |
| `el critico de teatro aburrido` | Guion | Humorístico |
| `la funcion que nunca termino` | Guion | Humorístico |

## Cómo mejorar el modelo

1. **Genera** un texto con cualquier combo
2. **Lee el resultado** — no esperes narrativa coherente aún (el modelo aprende palabra por palabra)
3. **Da feedback:**
   - 👍 si el vocabulario y estilo van en la dirección correcta
   - 👎 si no te gusta
4. **Repite** con distintas combinaciones
5. Después de **15-20 feedbacks** el modelo empezará a ajustarse a tus preferencias
6. **Los textos mejoran con el uso** — entre más feedback des, más aprende

## Concejo rápido

Empieza con **una sola combinación** (ej: Cuento/Misterioso) y dale feedback a 5-6 textos seguidos. El modelo converge más rápido si el feedback es consistente. Luego cambia a otra combinación.
