"""Write the resolved parameters as a sectioned ini file."""
import datetime

SECTION_ORDER = ["general", "input", "preprocessing", "replication", "mc", "md", "output"]

def write_config(args, path):
    lines = [f"; Resolved RepliSage config - generated {datetime.datetime.now().isoformat(timespec='seconds')}\n"]
    for sec in SECTION_ORDER:
        lines.append(f"\n[{sec}]\n")
        for a in args:
            if a.section != sec: continue
            lines.append(f"; {a.help}\n{a.name.lower()} = {'' if a.val is None else a.val}\n")
    with open(path, "w") as f: f.writelines(lines)
