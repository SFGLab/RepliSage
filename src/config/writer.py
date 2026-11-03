"""Write the parameters as a sectioned, commented ini file (template or resolved run)."""
import datetime

SECTION_ORDER = ["general", "input", "preprocessing", "replication", "mc", "md", "output"]
TITLES = {"general": "run, hardware", "input": "data and region", "preprocessing": "replication-timing preprocessing",
          "replication": "replication", "mc": "Monte Carlo and loop extrusion", "md": "3D structure (OpenMM)", "output": "report and figures"}
NOTES = {"BEDPE_PATH": "REQUIRED", "CHROM": "REQUIRED",
         "REPT_PATH": "file (.bigWig/.txt/.parquet) | builtin | none"}

def _fmt(v): return "" if v is None else v

def write_config(args, path, title="Resolved RepliSage config"):
    """One line per parameter, help as a comment above it (empty = not set)."""
    lines = [f"; {title} - generated {datetime.datetime.now().isoformat(timespec='seconds')}\n"]
    for sec in SECTION_ORDER:
        lines.append(f"\n[{sec}]   ; {TITLES[sec]}\n")
        for a in args:
            if a.section != sec: continue
            note = f" [{NOTES[a.name]}]" if a.name in NOTES else ""
            lines.append(f"; {a.help}{note}\n{a.name.lower()} = {_fmt(a.val)}\n")
    with open(path, "w") as f: f.writelines(lines)
