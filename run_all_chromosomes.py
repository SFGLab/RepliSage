"""Run RepliSage for every chromosome (whole-chromosome models) with one resolution.

    python run_all_chromosomes.py            # asks for the resolution in kb
Settings come from src/config.ini; only chromosome, bead number and run name are overridden.
"""
import math
import subprocess
import sys

# Chromosome sizes in bp (GRCh38)
chromosome_sizes = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555, "5": 181538259, "6": 170805979,
    "7": 159345973, "8": 145138636, "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345, "17": 83257441, "18": 80373285,
    "19": 58617616, "20": 64444167, "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}

def main():
    resolution = int(input("Enter resolution in kb: ")) * 1000
    for chrom, size in chromosome_sizes.items():
        n_beads = math.ceil(size / resolution)
        print(f'Running chromosome {chrom} with {n_beads} beads...')
        subprocess.run([sys.executable, "-m", "src.run", "-c", "src/config.ini",
                        "--chrom", f"chr{chrom}", "--region_start", "", "--region_end", "",   # empty = whole chromosome
                        "--n_beads", str(n_beads), "--n_lef", str(n_beads // 10), "--n_lef2", str(n_beads // 10),
                        "--run_tag", f"{resolution // 1000}kb"])

if __name__ == "__main__":
    main()
