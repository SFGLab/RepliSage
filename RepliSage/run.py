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
    """Prepare list of arguments.
    First, defaults are set.
    Then, optionally config file values.
    Finally, CLI arguments overwrite everything."""

    print("Reading config...")

    # Step 1: Setup argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--config_file', help="Specify config file (ini format)", metavar="FILE")

    for arg in args:
        arg_parser.add_argument(f"--{arg.name.lower()}", help=arg.help)

    args_ap = arg_parser.parse_args()  # parse command-line arguments
    args_dict = vars(args_ap)

    # Step 2: If config file provided, parse it
    if args_ap.config_file:
        config_parser = configparser.ConfigParser()
        config_parser.read(args_ap.config_file)
        args_cp = my_config_parser(config_parser)

        # Override default args with values from config file
        for cp_arg in args_cp:
            name, value = cp_arg
            arg = args.get_arg(name)
            arg.val = value

    # Step 3: Override again with CLI arguments (if present)
    for name, value in args_dict.items():
        if name == "config_file":
            continue
        if value is not None:
            arg = args.get_arg(name.upper())
            arg.val = value

    # Step 4: Finalize
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
    mode, rw, random_spins, p_rew, rep_fork_organizers = args.METHOD, args.LEF_RW, args.RANDOM_INIT_SPINS, args.P_REW, args.REP_FORK_EPIGENETIC_ORGANIZER
    Tstd_factor, speed_scale, init_rate_scale, p_rew = args.REP_T_STD_FACTOR, args.REP_SPEED_SCALE, args.REP_INIT_RATE_SCALE, args.P_REW
    save_MDT, save_plots, viz_heats = args.SAVE_MDT, args.SAVE_PLOTS, args.VIZ_HEATS
    
    # Define data and coordinates
    region, chrom =  [args.REGION_START, args.REGION_END], args.CHROM
    bedpe_file = args.BEDPE_PATH
    rept_path = args.REPT_PATH
    out_path = args.OUT_PATH
    
    # Run simulation
    sim = StochasticSimulation(N_beads, chrom, region, bedpe_file, out_path, N_lef, N_lef2, rept_path, t_rep, rep_duration, Tstd_factor, speed_scale, init_rate_scale)
    sim.run_stochastic_simulation(N_steps, MC_step, burnin, T, T_min, f, f2, b, kappa, c_rep, c_state_field, c_state_interact, mode, rw, p_rew, rep_fork_organizers)
    sim.run_openmm(args.PLATFORM,mode=args.SIMULATION_TYPE,init_struct=args.INITIAL_STRUCTURE_TYPE,integrator_mode=args.INTEGRATOR_TYPE,integrator_step=args.INTEGRATOR_STEP,p_ev=args.EV_P,sim_step=args.SIM_STEP,tol=args.TOLERANCE,md_temperature=args.SIM_TEMP,ff_path=args.FORCEFIELD_PATH)
    if save_plots:
        print('\nPloting stuff...') 
        sim.show_plots()
        sim.compute_structure_metrics()
        print('Done!')
    
    # Save Parameters
    if save_MDT:
        print('\nCreating metadata...')
        params = {k: v for k, v in locals().items() if k not in ['args','sim']} 
        save_parameters(out_path+'/other/params.txt',**params)
        print('Done')

    # Heatmap Visualization
    if args.VIZ_HEATS:
        print('\nMaking averaged heatmap plots...')
        print('Before replication...')
        get_avg_heatmap(args.OUT_PATH,1,(args.REP_START_TIME-args.BURNIN)//args.MC_STEP+1)
        print('During replication...')
        get_avg_heatmap(args.OUT_PATH,(args.REP_START_TIME-args.BURNIN)//args.MC_STEP+1,(args.REP_START_TIME+args.REP_TIME_DURATION-args.BURNIN)//args.MC_STEP+1)
        print('After replication...')
        get_avg_heatmap(args.OUT_PATH,(args.REP_START_TIME+args.REP_TIME_DURATION-args.BURNIN)//args.MC_STEP+1,(args.N_STEPS-args.BURNIN)//args.MC_STEP+1)
        print('And all of them together...')
        get_avg_heatmap(args.OUT_PATH,1,(args.N_STEPS-args.BURNIN)//args.MC_STEP+1)
        print('Done!')

if __name__=='__main__':
    main()
