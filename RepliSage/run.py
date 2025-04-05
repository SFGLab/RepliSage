from .stochastic_model import *
from .args_definition import *
import os
import time
import argparse
import configparser
from typing import List
from sys import stdout

def my_config_parser(config_parser: configparser.ConfigParser) -> List[tuple[str, str]]:
    """Helper function that makes flat list arg name, and it's value from ConfigParser object."""
    sections = config_parser.sections()
    all_nested_fields = [dict(config_parser[s]) for s in sections]
    args_cp = []
    for section_fields in all_nested_fields:
        for name, value in section_fields.items():
            args_cp.append((name, value))
    return args_cp

def get_config() -> ListOfArgs:
    """This function prepares the list of arguments.
    At first List of args with defaults is read.
    Then it's overwritten by args from config file (ini file).
    In the end config is overwritten by argparse options."""

    print(f"Reading config...")
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-c', '--config_file', help="Specify config file (ini format)", metavar="FILE")
    for arg in args:
        arg_parser.add_argument(f"--{arg.name.lower()}", help=arg.help)
    args_ap = arg_parser.parse_args()  # args from argparse
    config_parser = configparser.ConfigParser()
    config_parser.read(args_ap.config_file)
    args_cp = my_config_parser(config_parser)
    # Override defaults args with values from config file
    for cp_arg in args_cp:
        name, value = cp_arg
        arg = args.get_arg(name)
        arg.val = value
    # Now again override args with values from command line.
    for ap_arg in args_ap.__dict__:
        if ap_arg not in ['config_file']:
            name, value = ap_arg, getattr(args_ap, ap_arg)
            if value is not None:
                arg = args.get_arg(name)
                arg.val = value
    args.to_python()
    args.write_config_file()
    return args

def main():
    # Input arguments
    args = get_config()
    
    # Set parameters
    N_beads, N_lef, N_lef2 = args.N_BEADS, args.N_LEF, args.N_LEF2
    N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration = args.N_STEPS, args.MC_STEP, args.BURNIN, args.T_INIT, args.T_FINAL, args.REP_START_TIME, args.REP_TIME_DURATION
    f, f2, b, kappa = args.FOLDING_COEFF, args.FOLDING_COEFF2, args.BIND_COEFF, args.CROSS_COEFF
    c_state_field, c_state_interact, c_rep = args.POTTS_FIELD_COEFF, args.POTTS_INTERACT_COEFF, args.REP_COEFF
    mode, rw, random_spins = args.METHOD, args.LEF_RW, args.RANDOM_INIT_SPINS
    Tstd_factor, speed_scale, init_rate_scale = args.REP_T_STD_FACTOR, args.REP_SPEED_SCALE, args.REP_INIT_RATE_SCALE

    # for stress scale=5.0, sigma_t = T*0.2, speed=5*
    # for normal replication scale=1.0, sigma_t = T*0.1, speed=20*
    
    # Define data and coordinates
    # region, chrom =  [82835000, 98674700], 'chr14'
    region, chrom =  [args.REGION_START, args.REGION_END], args.CHROM
    bedpe_file = args.BEDPE_PATH
    rept_path = args.REPT_PATH
    out_path = args.OUT_PATH
    
    # Run simulation
    sim = StochasticSimulation(N_beads, chrom, region, bedpe_file, out_path, N_lef, N_lef2, rept_path, t_rep, rep_duration, Tstd_factor, speed_scale, init_rate_scale)
    sim.run_stochastic_simulation(N_steps, MC_step, burnin, T, T_min, f, f2, b, kappa, c_rep, c_state_field, c_state_interact, mode, rw)
    sim.show_plots()
    sim.run_openmm(args.PLATFORM,mode=args.SIMULATION_TYPE,init_struct=args.INITIAL_STRUCTURE_TYPE,integrator_mode=self.INTEGRATOR_STEP,p_ev=args.EV_P,tol=args.TOLERANCE,md_temperature=args.SIM_TEMP)
    sim.compute_structure_metrics()

    # Save Parameters
    params = {
        'N_beads': N_beads,
        'N_lef': N_lef,
        'N_lef2': N_lef2,
        'N_steps': N_steps,
        'MC_step': MC_step,
        'burnin': burnin,
        'T': T,
        'T_min': T_min,
        't_rep': t_rep,
        'rep_duration': rep_duration,
        'f': f,
        'f2': f2,
        'b': b,
        'kappa': kappa,
        'c_state_field': c_state_field,
        'c_state_interact': c_state_interact,
        'c_rep': c_rep,
        'mode': mode,
        'rw': rw,
        'random_spins': random_spins,
        'Tstd_factor': Tstd_factor,
        'speed_scale': speed_scale,
        'init_rate_scale': init_rate_scale,
        'region': region,
        'chrom': chrom,
        'bedpe_file': bedpe_file,
        'rept_path': rept_path,
        'out_path': out_path
    }
    save_parameters(out_path+'/other/params.txt',params)

if __name__=='__main__':
    main()