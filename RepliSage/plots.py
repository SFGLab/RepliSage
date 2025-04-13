import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from tqdm import tqdm
import time

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
    plt.savefig(out_path+'/plots/same_diff_sign.png',format='png',dpi=200)
    plt.savefig(out_path+'/plots/same_diff_sign.svg',format='svg',dpi=200)
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
    plt.savefig(out_path+'/plots/loop_length.png',format='svg',dpi=200)
    plt.savefig(out_path+'/plots/loop_length.svg',format='svg',dpi=200)
    plt.grid(True)
    plt.close()

def coh_traj_plot(ms,ns,N_beads,path):
    print('\nPlotting trajectories of cohesins...')
    start = time.time()
    N_coh = len(ms)
    figure(figsize=(20, 20),dpi=400)
    color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) for i in range(N_coh)]
    size = 0.1
    
    for nn in tqdm(range(N_coh)):
        tr_m, tr_n = ms[nn], ns[nn]
        plt.fill_between(np.arange(len(tr_m)), tr_m, tr_n, color=color[nn], alpha=0.4, interpolate=False, linewidth=0)
    plt.xlabel('Simulation Step', fontsize=28)
    plt.ylabel('Position of Cohesin', fontsize=28)
    plt.gca().invert_yaxis()
    plt.ylim((0,N_beads))
    # plt.gca().set_ylim(bottom=0) 
    save_path = path+'/plots/LEFs.png'
    plt.savefig(save_path,format='png')
    save_path = path+'/plots/LEFs.svg'
    plt.savefig(save_path,format='svg')
    plt.close()
    end = time.time()
    elapsed = end - start
    print(f'Plot created succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

def make_timeplots(Es, Es_ising, Fs, Bs, Rs, mags, burnin, path=None):
    figure(figsize=(10, 6), dpi=200)
    plt.plot(Es, 'black',label='Total Energy')
    plt.plot(Es_ising, 'orange',label='Potts Energy')
    plt.plot(Fs, 'b',label='Folding Energy')
    plt.plot(Bs, 'r',label='Binding Energy')
    # plt.plot(Rs, 'g',label='Replication Energy')
    plt.ylabel('Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    # plt.yscale('symlog')
    plt.legend()
    save_path = path+'/plots/energies.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/energies.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/energies.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    figure(figsize=(10, 6), dpi=200)
    plt.plot(Es, 'k',label='Total Energy')
    plt.ylabel('Total Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/total_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/total_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/total_energy.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    figure(figsize=(10, 6), dpi=200)
    plt.plot(mags, 'purple',label='mags')
    plt.ylabel('Magnetization', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/mag.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/mag.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/mag.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    figure(figsize=(10, 6), dpi=200)
    plt.plot(Fs, 'b')
    plt.ylabel('Folding Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/fold_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/fold_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/fold_energy.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    figure(figsize=(10, 6), dpi=200)
    plt.plot(Es_ising, 'orange')
    plt.ylabel('Energy of the Potts Model', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/potts_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/potts_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/potts_energy.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    figure(figsize=(10, 6), dpi=200)
    plt.plot(Rs, 'g')
    plt.ylabel('Replication Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/repli_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/repli_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    save_path = path+'/plots/repli_energy.png'
    plt.savefig(save_path,format='png',dpi=200)
    plt.close()
    
    # Step 1: Fit regression model
    ys = np.array(Fs)[burnin:]
    xs = np.arange(len(ys))
    coeffs = np.polyfit(xs, ys, 6)  # Polynomial coefficients
    trend = np.polyval(coeffs, xs)  # Evaluate the polynomial at x

    # Step 2: Detrend the signal
    detrended_signal = ys - trend

    figure(figsize=(10, 6), dpi=400)
    plot_acf(detrended_signal, title=None, lags = len(np.array(Fs)[burnin:])//2)
    plt.ylabel("Autocorrelations", fontsize=16)
    plt.xlabel("Lags", fontsize=16)
    plt.grid()
    if path!=None:
        save_path = path+'/plots/autoc.png'
        plt.savefig(save_path,dpi=400)
        save_path = path+'/plots/autoc.svg'
        plt.savefig(save_path,format='svg',dpi=200)
        save_path = path+'/plots/autoc.pdf'
        plt.savefig(save_path,format='pdf',dpi=200)
        save_path = path+'/plots/autoc.png'
        plt.savefig(save_path,format='png',dpi=200)
    plt.close()

def ising_traj_plot(traj,save_path):
    figure(figsize=(20, 20),dpi=500)
    plt.imshow(traj,cmap='rainbow',aspect='auto')
    plt.xlabel('Computational Time',fontsize=28)
    plt.ylabel('Region', fontsize=28)
    plt.savefig(save_path+'/plots/potts_traj.png',format='png',dpi=400)
    plt.savefig(save_path+'/plots/potts_traj.svg',format='svg',dpi=400)
    plt.close()
