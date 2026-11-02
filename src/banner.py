"""RepliSage ASCII banner and a pool of science quotes."""
import random

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

# (quote, author) -- short, real, and relevant to modelling
QUOTES = [
    ("All models are wrong, but some are useful.", "George Box"),
    ("What I cannot create, I do not understand.", "Richard Feynman"),
    ("Nothing in biology makes sense except in the light of evolution.", "T. Dobzhansky"),
    ("Science is built up of facts, as a house is built of stones.", "Henri Poincare"),
    ("The cell is the unit of life; the loop is its handwriting.", "RepliSage"),
    ("Loops extrude fast, forks move slow; chromatin negotiates.", "RepliSage"),
]

def get_quote():
    q, a = random.choice(QUOTES)
    return f'"{q}"  - {a}'
