"""Argument container and type casting for RepliSage parameters."""
from dataclasses import dataclass
from typing import Any

@dataclass
class Arg:
    section: str      # section in config.ini
    name: str         # UPPERCASE parameter name (case-insensitive in ini/CLI)
    help: str
    type: type
    default: Any
    val: Any = None

    def __post_init__(self):
        if self.val is None: self.val = self.default

TRUE, FALSE = {"true", "1", "y", "yes"}, {"false", "0", "n", "no"}

def cast(arg: Arg):
    """Cast the string value of an argument to its python type ('' -> None)."""
    v = arg.val
    if v is None or (isinstance(v, str) and v.strip() == ""): return None
    if arg.type is bool:
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        if s in TRUE: return True
        if s in FALSE: return False
        raise ValueError(f"Can't convert {arg.name} = {v!r} into bool.")
    if arg.type is str: return str(v).strip()
    return arg.type(float(v)) if arg.type is int and isinstance(v, str) and "." in v else arg.type(v)

class Args:
    """List of Arg objects with attribute access: args.N_BEADS."""
    def __init__(self, arg_list):
        self._args = {a.name: a for a in arg_list}

    def get_arg(self, name):
        try: return self._args[name.upper()]
        except KeyError: raise ValueError(f"No such arg: {name.upper()}")

    def __getattr__(self, item):
        if item.startswith("_"): raise AttributeError(item)
        return self.get_arg(item).val

    def set(self, name, value):
        self.get_arg(name).val = value

    def has(self, name):
        return name.upper() in self._args

    def __iter__(self):
        return iter(self._args.values())

    def to_python(self):
        for a in self: a.val = cast(a)

    def as_dict(self):
        return {a.name: a.val for a in self}
