import configparser
import subprocess
import math
import os

# Chromosome sizes in bp (GRCh38)
chromosome_sizes = {
    "1": 248956422,
    "2": 242193529,
    "3": 198295559,
    "4": 190214555,
    "5": 181538259,
    "6": 170805979,
    "7": 159345973,
    "8": 145138636,
    "9": 138394717,
    "10": 133797422,
    "11": 135086622,
    "12": 133275309,
    "13": 114364328,
    "14": 107043718,
    "15": 101991189,
    "16": 90338345,
    "17": 83257441,
    "18": 80373285,
    "19": 58617616,
    "20": 64444167,
    "21": 46709983,
    "22": 50818468,
    "X": 156040895,
    "Y": 57227415,
}

config_file = 'config.ini'
def main():
    resolution_kb = int(input("Enter resolution in kb: "))
    resolution = resolution_kb * 1000

    for chrom, size in chromosome_sizes.items():
        n_beads = math.ceil(size / resolution)
        
        # Load config
        config = configparser.ConfigParser()
        config.read(config_file)

        # Ensure section exists
        if 'Main' not in config:
            config['Main'] = {}

        config['Main']['N_BEADS'] = str(n_beads)
        config['Main']['N_LEF'] = str(n_beads//10)
        config['Main']['N_LEF2'] = str(n_beads//10)
        config['Main']['CHROM'] = 'chr'+chrom
        config['Main']['REGION_START'] = str(0)
        config['Main']['REGION_END'] = str(size)
        config['Main']['OUT_PATH'] = f'tmp/chrom{chrom}_ht0'

        # Write back updated config.ini
        with open(config_file, 'w') as configfile:
            config.write(configfile)

        # Run command
        print(f'Running chromosome {chrom} with {n_beads} beads...')
        subprocess.run(["python", "-m", "RepliSage.run", "-c", config_file])

if __name__ == "__main__":
    main()
