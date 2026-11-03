"""RepliSage ASCII banner and a pool of science quotes."""
import random
import textwrap

try:
    from ._version import __version__ as VERSION
except ImportError:  # script run
    from _version import __version__ as VERSION

ART = r"""
 ____            _ _ ____
|  _ \ ___ _ __ | (_) ___|  __ _  __ _  ___
| |_) / _ \ '_ \| | \___ \ / _` |/ _` |/ _ \
|  _ <  __/ |_) | | |___) | (_| | (_| |  __/
|_| \_\___| .__/|_|_|____/ \__,_|\__, |\___|
          |_|                     |___/
"""

# (quote, author): short and relevant to modelling; the ancient ones are traditional attributions
QUOTES = [
    # modelling and science
    ("All models are wrong, but some are useful.", "George Box"),
    ("What I cannot create, I do not understand.", "Richard Feynman"),
    ("Nothing in biology makes sense except in the light of evolution.", "Theodosius Dobzhansky"),
    ("Science is built up of facts, as a house is built of stones.", "Henri Poincare"),
    ("Everything should be made as simple as possible, but not simpler.", "Albert Einstein"),
    # physics, fantasy, poetry
    ("The first principle is that you must not fool yourself - and you are the easiest person to fool.", "Richard Feynman"),
    ("Nature uses only the longest threads to weave her patterns, so each small piece of her fabric reveals the organization of the entire tapestry.", "Richard Feynman"),
    ("Not all those who wander are lost.", "J. R. R. Tolkien"),
    ("It's a dangerous business, Frodo, going out your door. You step onto the road, and there's no knowing where you might be swept off to.", "J. R. R. Tolkien"),
    ("We shall not cease from exploration, and the end of all our exploring will be to arrive where we started and know the place for the first time.", "T. S. Eliot"),
    ("Where is the wisdom we have lost in knowledge? Where is the knowledge we have lost in information?", "T. S. Eliot"),
    ("I hope for nothing. I fear nothing. I am free.", "Nikos Kazantzakis"),
    # Greece
    ("Give me a place to stand, and I shall move the Earth.", "Archimedes"),
    ("Nothing exists except atoms and empty space; everything else is opinion.", "Democritus"),
    ("Number rules the universe.", "Pythagoras"),
    ("The whole is greater than the sum of its parts.", "Aristotle"),
    ("Nothing endures but change.", "Heraclitus"),
    ("Life is short, the art is long.", "Hippocrates"),
    ("Wonder is the feeling of a philosopher, and philosophy begins in wonder.", "Plato"),
    # Poland
    ("Nothing in life is to be feared, it is only to be understood.", "Marie Sklodowska-Curie"),
    ("I was taught that the way of progress was neither swift nor easy.", "Marie Sklodowska-Curie"),
    ("To know that we know what we know, and that we do not know what we do not know, that is true knowledge.", "Nicolaus Copernicus"),
    ("Good mathematicians see analogies between theorems; great ones see analogies between analogies.", "Stefan Banach"),
    ("Whatever inspiration is, it is born from a continuous 'I don't know'.", "Wislawa Szymborska"),
    # the model itself
    ("The cell is the unit of life; the loop is its handwriting.", "RepliSage"),
    ("Loops extrude fast, forks move slow; chromatin negotiates.", "RepliSage"),
]

_rng = random.SystemRandom()   # independent of the simulation seed, so the quote changes at every run

def get_quote():
    """A different quote at (almost) every run."""
    q, a = _rng.choice(QUOTES)
    return textwrap.fill(f'"{q}"  - {a}', width=76, subsequent_indent="    ")
