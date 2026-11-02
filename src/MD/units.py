"""Parse strings such as '100 femtosecond' or '310 kelvin' into OpenMM quantities."""
import re
import openmm as mm
from openmm.unit import Quantity

_RX = re.compile(r'(?P<value>[-+]?\d+(?:\.\d+)?) ?(?P<unit>\w+)')

def parse_quantity(val):
    """'310 kelvin' -> Quantity(310, kelvin). Quantities pass through unchanged."""
    if isinstance(val, Quantity): return val
    m = _RX.match(str(val))
    if not m: raise ValueError(f"Can't parse quantity '{val}'. Example: 12.3 femtosecond.")
    value, unit = m.groups()
    try: unit = getattr(mm.unit, unit)
    except AttributeError: raise ValueError(f"Unknown unit '{unit}' in '{val}'.")
    return Quantity(value=float(value), unit=unit)
