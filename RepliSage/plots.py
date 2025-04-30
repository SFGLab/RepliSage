import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from tqdm import tqdm
import time
from scipy.signal import detrend

def compute_state_proportions_sign_based(Ms, Ns, Cs, S_time, G2_time, out_path=None):
    """
    Computes the proportion of links where connected nodes are:
    - in the same sign state (both positive or both negative)
    - in different sign states
    as a function of time.

    Args:
        Ms: (array) Source node indices [i, t]
        Ns: (array) Target node indices [i, t]
        Cs: (array) Node states [n, t]
    
    Returns:
        same_sign_fraction: array of proportion of same-sign links at each time
        diff_sign_fraction: array of proportion of different-sign links at each time
    """
    num_times = Ms.shape[1]
    same_sign_fraction = np.zeros(num_times)
    diff_sign_fraction = np.zeros(num_times)

    for t in range(num_times):
        m_nodes = Ms[:, t]
        n_nodes = Ns[:, t]
        
        valid = (m_nodes >= 0) & (n_nodes >= 0)

        if np.sum(valid) == 0:
            continue
        
        m_states = Cs[m_nodes[valid], t]
        n_states = Cs[n_nodes[valid], t]
        
        # Check if one is positive and the other negative
        different_sign = (m_states > 0) & (n_states < 0) | (m_states < 0) & (n_states > 0)
        
        same_sign = ~different_sign  # complement
        
        same_sign_fraction[t] = np.sum(same_sign) / np.sum(valid)
        diff_sign_fraction[t] = np.sum(different_sign) / np.sum(valid)

    plt.figure(figsize=(10, 6),dpi=200)
    times = np.arange(len(same_sign_fraction))

    plt.plot(times, same_sign_fraction, label='Same State Links',color='red')
    plt.plot(times, diff_sign_fraction, label='Different State Links',color='blue')
    plt.xlabel('MC step',fontsize=16)
    plt.ylabel('Proportion',fontsize=16)
    plt.legend()

    # Vertical line at x = 123
    plt.axvline(x=S_time, color='red', linestyle='--', label='x = 123')

    # # Annotate G1 phase
    # plt.annotate('G1 phase', 
    #             xy=(S_time-50, 0.38),  # Position of the annotation (centered)
    #             xytext=(S_time-50, 0.38),  # Text position
    #             fontsize=14)

    # Vertical line at x = 123
    plt.axvline(x=G2_time, color='red', linestyle='--', label='x = 123')

    # # Annotate G1 phase
    # plt.annotate('S phase', 
    #             xy=(S_time+50, 0.38),  # Position of the annotation (centered)
    #             xytext=(S_time+50, 0.38),  # Text position
    #             fontsize=14)

    # # Annotate G1 phase
    # plt.annotate('G2/M phase', 
    #             xy=(G2_time+50, 0.42),  # Position of the annotation (centered)
    #             xytext=(G2_time+50, 0.5),  # Text position
    #             fontsize=14)

    # plt.ylim((0,1))
    # plt.title('Proportion of Same-State and Different-State Links Over Time')
    plt.savefig(out_path+'/plots/graph_metrics/same_diff_sign.png',format='png',dpi=200)
    plt.savefig(out_path+'/plots/graph_metrics/same_diff_sign.svg',format='svg',dpi=200)
    plt.grid(True)
    plt.close()

    return same_sign_fraction, diff_sign_fraction

def plot_loop_length(Ls, S_time, G2_time, out_path=None):
    """
    Plots how the probability distribution changes over columns of matrix Ls using plt.imshow.
    
    Parameters:
        Ls (np.ndarray): 2D array where rows represent samples, and columns represent time points.
        out_path (str, optional): Path to save the heatmap. If None, it will only display the plot.
    """
    avg_Ls = np.average(Ls,axis=0)
    std_Ls = np.std(Ls,axis=0)
    sem_Ls = std_Ls / np.sqrt(Ls.shape[0])  # SEM = std / sqrt(N)
    ci95 = 1.96 * sem_Ls

    # Plot
    plt.figure(figsize=(10, 6),dpi=200)
    x = np.arange(len(avg_Ls))
    plt.plot(x, avg_Ls, label='Average Ls')
    plt.fill_between(x, avg_Ls - ci95, avg_Ls + ci95, alpha=0.2, label='Confidence Interval (95%)')
    plt.xlabel('MC step',fontsize=16)
    plt.ylabel('Average Loop Length',fontsize=16)
    plt.legend()
    # Vertical line at x = 123
    plt.axvline(x=S_time, color='red', linestyle='--', label='x = 123')

    # # Annotate G1 phase
    # plt.annotate('G1 phase', 
    #             xy=(S_time-50, 0.38),  # Position of the annotation (centered)
    #             xytext=(S_time-50, 0.38),  # Text position
    #             fontsize=14)

    # Vertical line at x = 123
    plt.axvline(x=G2_time, color='red', linestyle='--', label='x = 123')

    # # Annotate G1 phase
    # plt.annotate('S phase', 
    #             xy=(S_time+50, 0.38),  # Position of the annotation (centered)
    #             xytext=(S_time+50, 0.38),  # Text position
    #             fontsize=14)

    # # Annotate G1 phase
    # plt.annotate('G2/M phase', 
    #             xy=(G2_time+50, 0.42),  # Position of the annotation (centered)
    #             xytext=(G2_time+50, 0.5),  # Text position
    #             fontsize=14)

    # plt.title('Average Ls with 95% Confidence Interval',fontsize=16)
    plt.savefig(out_path+'/plots/MCMC_diagnostics/loop_length.png',format='svg',dpi=200)
    plt.savefig(out_path+'/plots/MCMC_diagnostics/loop_length.svg',format='svg',dpi=200)
    plt.grid(True)
    plt.close()

def coh_traj_plot(ms, ns, N_beads, path, jump_threshold=200, min_stable_time=10):
    print('\nPlotting trajectories of cohesins...')
    start = time.time()
    N_coh = len(ms)
    figure(figsize=(10, 10), dpi=200)
    cmap = plt.get_cmap('prism')
    colors = [cmap(i / N_coh) for i in range(N_coh)]

    for nn in tqdm(range(N_coh)):
        tr_m, tr_n = np.array(ms[nn]), np.array(ns[nn])
        steps = np.arange(len(tr_m))

        # Calculate jump size for tr_m and tr_n independently
        jumps_m = np.abs(np.diff(tr_m))
        jumps_n = np.abs(np.diff(tr_n))

        # Create mask: True = good point, False = jump
        jump_mask = np.ones_like(tr_m, dtype=bool)
        jump_mask[1:] = (jumps_m < jump_threshold) & (jumps_n < jump_threshold)  # both must be below threshold

        # Now we want to detect stable regions
        stable_mask = np.copy(jump_mask)

        # Find connected regions
        current_length = 0
        for i in range(len(stable_mask)):
            if jump_mask[i]:
                current_length += 1
            else:
                if current_length < min_stable_time:
                    stable_mask[i-current_length:i] = False
                current_length = 0
        # Handle last region
        if current_length < min_stable_time:
            stable_mask[len(stable_mask)-current_length:] = False

        # Apply mask
        tr_m_masked = np.ma.masked_array(tr_m, mask=~stable_mask)
        tr_n_masked = np.ma.masked_array(tr_n, mask=~stable_mask)

        plt.fill_between(steps, tr_m_masked, tr_n_masked,
                         color=colors[nn], alpha=0.6, interpolate=False, linewidth=0)
    plt.xlabel('MC Step', fontsize=16)
    plt.ylabel('Simulation Beads', fontsize=16)
    plt.gca().invert_yaxis()
    plt.ylim((0, N_beads))
    save_path = path + '/plots/MCMC_diagnostics/LEFs.png'
    plt.savefig(save_path, format='png')
    save_path = path + '/plots/MCMC_diagnostics/LEFs.svg'
    plt.savefig(save_path, format='svg')
    plt.close()
    end = time.time()
    elapsed = end - start
    print(f'Plot created successfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and {elapsed%60:.0f} seconds.')

def make_timeplots(Es, Es_potts, Fs, Bs, mags, burnin, path=None):
    figure(figsize=(10, 6), dpi=200)
    # plt.plot(Es, 'black',label='Total Energy')
    plt.plot(Es_potts, 'orange',label='Potts Energy')
    plt.plot(Fs, 'b',label='Folding Energy')
    plt.plot(Bs, 'r',label='Binding Energy')
    # plt.plot(Rs, 'g',label='Replication Energy')
    plt.ylabel('Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    # plt.yscale('symlog')
    plt.legend()
    save_path = path+'/plots/MCMC_diagnostics/energies.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/energies.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/energies.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6),dpi=200)
    plt.plot(Es, 'k',label='Total Energy')
    plt.ylabel('Total Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/MCMC_diagnostics/total_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/total_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/total_energy.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6),dpi=200)
    plt.plot(mags, 'purple',label='mags')
    plt.ylabel('Magnetization', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/MCMC_diagnostics/mag.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/mag.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/mag.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6),dpi=200)
    plt.plot(Fs, 'b')
    plt.ylabel('Folding Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/MCMC_diagnostics/fold_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/fold_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/fold_energy.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6),dpi=200)
    plt.plot(Es_potts, 'orange')
    plt.ylabel('Energy of the Potts Model', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/MCMC_diagnostics/potts_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/potts_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/potts_energy.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6),dpi=200)
    plt.plot(Bs, 'g')
    plt.ylabel('Binding Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/MCMC_diagnostics/bind_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/bind_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/MCMC_diagnostics/bind_energy.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()
    
    # Step 1: Use a non-parametric method to remove the trend

    ys = np.array(Fs)[burnin:]
    detrended_signal = detrend(ys, type='linear')  # Remove linear trend

    # Step 2: Plot the autocorrelation of the detrended signal
    figure(figsize=(10, 6), dpi=400)
    plot_acf(detrended_signal, title=None, lags=len(detrended_signal) // 2)
    plt.ylabel("Autocorrelations", fontsize=16)
    plt.xlabel("Lags", fontsize=16)
    plt.grid()
    if path is not None:
        save_path = path + '/plots/MCMC_diagnostics/autoc.png'
        plt.savefig(save_path, dpi=400)
        save_path = path + '/plots/MCMC_diagnostics/autoc.svg'
        plt.savefig(save_path, format='svg', dpi=200)
        save_path = path + '/plots/MCMC_diagnostics/autoc.pdf'
        plt.savefig(save_path, format='pdf', dpi=200)
        save_path = path + '/plots/MCMC_diagnostics/autoc.png'
        plt.savefig(save_path, format='png', dpi=200)
    plt.close()

def ising_traj_plot(spins, save_path):
    plt.figure(figsize=(10, 10),dpi=200)
    plt.imshow(spins, cmap='bwr', aspect='auto')
    plt.xlabel('MC step', fontsize=16)
    plt.ylabel('Simulation Beads', fontsize=16)
    plt.savefig(save_path + '/plots/MCMC_diagnostics/potts_traj.png', format='png', dpi=200)
    plt.savefig(save_path + '/plots/MCMC_diagnostics/potts_traj.svg', format='svg', dpi=200)
    plt.close()