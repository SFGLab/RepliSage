import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
import time
from matplotlib.colors import TwoSlopeNorm
from statsmodels.tsa.stattools import acf

def compute_state_proportions_sign_based(Ms, Ns, Cs, burnin, S_time, G2_time, out_path=None, max_lag=50):
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

    plt.axvline(x=S_time, color='red', linestyle='--', label='S phase')
    plt.axvline(x=G2_time, color='red', linestyle='--', label='G2 phase')
    plt.savefig(out_path+'/plots/graph_metrics/same_diff_sign.png',format='png',dpi=200)
    plt.savefig(out_path+'/plots/graph_metrics/same_diff_sign.svg',format='svg',dpi=200)
    plt.grid(True)
    plt.close()

    # Violin plot for distribution of same_sign_fraction during different phases
    g1 = same_sign_fraction[:S_time]
    s = same_sign_fraction[S_time:G2_time]
    g2 = same_sign_fraction[G2_time:]

    data = [g1, s, g2]
    labels = ['G1', 'S', 'G2']

    plt.figure(figsize=(8, 6), dpi=200)
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks([1, 2, 3], labels, fontsize=14)
    plt.ylabel('Same Sign Fraction', fontsize=16)
    plt.title('Distribution of Same Sign Fraction by Cell Cycle Phase', fontsize=16)
    plt.grid(True, axis='y')
    if out_path:
        plt.savefig(out_path + '/plots/graph_metrics/same_sign_fraction_violin.png', format='png', dpi=200)
        plt.savefig(out_path + '/plots/graph_metrics/same_sign_fraction_violin.svg', format='svg', dpi=200)
    plt.close()

    # Compute autocorrelations for each phase of same_sign_fraction
    same_sign_fraction = np.asarray(same_sign_fraction)
    g1 = same_sign_fraction[burnin:S_time]
    s = same_sign_fraction[S_time:G2_time]
    g2 = same_sign_fraction[G2_time:]

    g1_acf = acf(g1, nlags=max_lag, fft=True)
    s_acf = acf(s, nlags=max_lag, fft=True)
    g2_acf = acf(g2, nlags=max_lag, fft=True)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=200)
    axs[0].bar(np.arange(len(g1_acf)), g1_acf, color="#5E81AC", alpha=0.6, label='Autocorr')
    axs[0].plot(np.arange(len(g1_acf)), g1_acf, color="#5E81AC", linewidth=2, label='Envelope')
    axs[0].set_title('Same Sign Fraction Autocorrelation (G1 phase)', fontsize=14)
    axs[0].set_ylabel('Autocorrelation', fontsize=12)
    axs[0].grid(True)

    axs[1].bar(np.arange(len(s_acf)), s_acf, color="#A3BE8C", alpha=0.6, label='Autocorr')
    axs[1].plot(np.arange(len(s_acf)), s_acf, color="#A3BE8C", linewidth=2, label='Envelope')
    axs[1].set_title('Same Sign Fraction Autocorrelation (S phase)', fontsize=14)
    axs[1].set_ylabel('Autocorrelation', fontsize=12)
    axs[1].grid(True)

    axs[2].bar(np.arange(len(g2_acf)), g2_acf, color="#D08770", alpha=0.6, label='Autocorr')
    axs[2].plot(np.arange(len(g2_acf)), g2_acf, color="#D08770", linewidth=2, label='Envelope')
    axs[2].set_title('Same Sign Fraction Autocorrelation (G2 phase)', fontsize=14)
    axs[2].set_xlabel('Lag', fontsize=12)
    axs[2].set_ylabel('Autocorrelation', fontsize=12)
    axs[2].grid(True)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path + '/plots/graph_metrics/same_sign_fraction_autocorr.png', format='png', dpi=200)
        plt.savefig(out_path + '/plots/graph_metrics/same_sign_fraction_autocorr.svg', format='svg', dpi=200)
    plt.close()

    return same_sign_fraction, diff_sign_fraction

def plot_loop_length(Ls, burnin, S_time, G2_time, out_path=None, max_lag=50):
    """
    Plots how the probability distribution changes over columns of matrix Ls using plt.imshow.
    Also creates violin plots for each cell cycle phase (G1, S, G2).
    
    Parameters:
        Ls (np.ndarray): 2D array where rows represent samples, and columns represent time points.
        S_time (int): Index where S phase starts.
        G2_time (int): Index where G2 phase starts.
        out_path (str, optional): Path to save the plots. If None, it will only display the plot.
    """
    avg_Ls = np.average(Ls, axis=0)
    std_Ls = np.std(Ls, axis=0)
    sem_Ls = std_Ls / np.sqrt(Ls.shape[0])  # SEM = std / sqrt(N)
    ci95 = 1.96 * sem_Ls

    # Line plot with confidence interval
    plt.figure(figsize=(10, 6), dpi=200)
    x = np.arange(len(avg_Ls))
    plt.plot(x, avg_Ls, label='Average Ls')
    plt.fill_between(x, avg_Ls - ci95, avg_Ls + ci95, alpha=0.2, label='Confidence Interval (95%)')
    plt.xlabel('MC step', fontsize=16)
    plt.ylabel('Average Loop Length', fontsize=16)
    plt.legend()
    plt.axvline(x=S_time, color='red', linestyle='--', label='S phase')
    plt.axvline(x=G2_time, color='red', linestyle='--', label='G2 phase')
    plt.grid(True)
    if out_path:
        plt.savefig(out_path + '/plots/MCMC_diagnostics/loop_length.png', format='png', dpi=200)
        plt.savefig(out_path + '/plots/MCMC_diagnostics/loop_length.svg', format='svg', dpi=200)
    plt.close()

    # Violin plots for each phase
    # G1: [0, S_time), S: [S_time, G2_time), G2: [G2_time, end)
    g1 = Ls[50:, :S_time].flatten()
    s = Ls[:, S_time:G2_time].flatten()
    g2 = Ls[:, G2_time:].flatten()

    data = [g1, s, g2]
    labels = ['G1', 'S', 'G2']

    plt.figure(figsize=(8, 6), dpi=200)
    parts = plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks([1, 2, 3], labels, fontsize=14)
    plt.ylabel('Loop Length', fontsize=16)
    plt.title('Loop Length Distribution by Cell Cycle Phase', fontsize=16)
    plt.grid(True, axis='y', which='both')
    if out_path:
        plt.savefig(out_path + '/plots/MCMC_diagnostics/loop_length_violin.png', format='png', dpi=200)
        plt.savefig(out_path + '/plots/MCMC_diagnostics/loop_length_violin.svg', format='svg', dpi=200)
    plt.close()

    # Compute autocorrelations for average loop length in each phase
    avg_Ls = np.asarray(avg_Ls)
    g1_avg = avg_Ls[burnin:S_time]
    s_avg = avg_Ls[S_time:G2_time]
    g2_avg = avg_Ls[G2_time:]

    g1_acf = acf(g1_avg, nlags=max_lag, fft=True)
    s_acf = acf(s_avg, nlags=max_lag, fft=True)
    g2_acf = acf(g2_avg, nlags=max_lag, fft=True)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=200)
    axs[0].bar(np.arange(len(g1_acf)), g1_acf, color="#5E81AC", alpha=0.6, label='Autocorr')
    axs[0].plot(np.arange(len(g1_acf)), g1_acf, color="#5E81AC", linewidth=2, label='Envelope')
    axs[0].set_title('Average Loop Length Autocorrelation (G1 phase)', fontsize=14)
    axs[0].set_ylabel('Autocorrelation', fontsize=12)
    axs[0].grid(True)

    axs[1].bar(np.arange(len(s_acf)), s_acf, color="#A3BE8C", alpha=0.6, label='Autocorr')
    axs[1].plot(np.arange(len(s_acf)), s_acf, color="#A3BE8C", linewidth=2, label='Envelope')
    axs[1].set_title('Average Loop Length Autocorrelation (S phase)', fontsize=14)
    axs[1].set_ylabel('Autocorrelation', fontsize=12)
    axs[1].grid(True)

    axs[2].bar(np.arange(len(g2_acf)), g2_acf, color="#D08770", alpha=0.6, label='Autocorr')
    axs[2].plot(np.arange(len(g2_acf)), g2_acf, color="#D08770", linewidth=2, label='Envelope')
    axs[2].set_title('Average Loop Length Autocorrelation (G2 phase)', fontsize=14)
    axs[2].set_xlabel('Lag', fontsize=12)
    axs[2].set_ylabel('Autocorrelation', fontsize=12)
    axs[2].grid(True)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path + '/plots/MCMC_diagnostics/loop_length_autocorr.png', format='png', dpi=200)
        plt.savefig(out_path + '/plots/MCMC_diagnostics/loop_length_autocorr.svg', format='svg', dpi=200)
    plt.close()

def coh_traj_plot(ms, ns, N_beads, path, jump_threshold=200, min_stable_time=10):
    """
    Plots the trajectories of cohesins as filled regions between their two ends over time.

    Parameters:
        ms (list of arrays): List where each element is an array of left-end positions of a cohesin over time.
        ns (list of arrays): List where each element is an array of right-end positions of a cohesin over time.
        N_beads (int): Total number of beads (simulation sites) in the system.
        path (str): Directory path where the plots will be saved.
        jump_threshold (int, optional): Maximum allowed jump (in bead units) between consecutive time points for both ends.
            If the jump between two consecutive positions exceeds this threshold for either end, that segment is considered a jump and is masked out.
            Lower values make the criterion for erasing (masking) trajectories more strict (more segments are erased), higher values make it less strict.
        min_stable_time (int, optional): Minimum number of consecutive time points required for a region to be considered stable and shown.
            Shorter stable regions (less than this value) are erased (masked out).
            Higher values make the criterion more strict (only longer stable regions are shown), lower values make it less strict.

    The function highlights only stable, non-jumping regions of cohesin trajectories.
    """
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

def make_timeplots(Es, Es_potts, Fs, Bs, mags, burnin, G1_end, G2M_end, path=None, max_lag=50):
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

    # Compute autocorrelations for each phase
    Fs = np.asarray(Fs)
    g1_Fs = Fs[burnin:G1_end]
    s_Fs = Fs[G1_end:G2M_end]
    g2m_Fs = Fs[G2M_end:]

    g1_acf = acf(g1_Fs, nlags=max_lag, fft=True)
    s_acf = acf(s_Fs, nlags=max_lag, fft=True)
    g2m_acf = acf(g2m_Fs, nlags=max_lag, fft=True)

    # Folding Energy Autocorrelation: Barplot with envelope
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=200)
    axs[0].bar(np.arange(len(g1_acf)), g1_acf, color="#5E81AC", alpha=0.6, label='Autocorr')
    axs[0].plot(np.arange(len(g1_acf)), g1_acf, color="#5E81AC", linewidth=2, label='Envelope')
    axs[0].set_title('Folding Energy Autocorrelation (G1 phase)', fontsize=14)
    axs[0].set_ylabel('Autocorrelation', fontsize=12)
    axs[0].grid(True)

    axs[1].bar(np.arange(len(s_acf)), s_acf, color="#A3BE8C", alpha=0.6, label='Autocorr')
    axs[1].plot(np.arange(len(s_acf)), s_acf, color="#A3BE8C", linewidth=2, label='Envelope')
    axs[1].set_title('Folding Energy Autocorrelation (S phase)', fontsize=14)
    axs[1].set_ylabel('Autocorrelation', fontsize=12)
    axs[1].grid(True)

    axs[2].bar(np.arange(len(g2m_acf)), g2m_acf, color="#D08770", alpha=0.6, label='Autocorr')
    axs[2].plot(np.arange(len(g2m_acf)), g2m_acf, color="#D08770", linewidth=2, label='Envelope')
    axs[2].set_title('Folding Energy Autocorrelation (G2M phase)', fontsize=14)
    axs[2].set_xlabel('Lag', fontsize=12)
    axs[2].set_ylabel('Autocorrelation', fontsize=12)
    axs[2].grid(True)

    plt.tight_layout()
    if path:
        plt.savefig(path + '/plots/MCMC_diagnostics/fold_energy_autocorr.png', format='png', dpi=200)
        plt.savefig(path + '/plots/MCMC_diagnostics/fold_energy_autocorr.svg', format='svg', dpi=200)
    plt.close()

    # Compute autocorrelations for Potts energy in each phase
    Es_potts = np.asarray(Es_potts)
    g1_potts = Es_potts[burnin:G1_end]
    s_potts = Es_potts[G1_end:G2M_end]
    g2m_potts = Es_potts[G2M_end:]

    g1_potts_acf = acf(g1_potts, nlags=max_lag, fft=True)
    s_potts_acf = acf(s_potts, nlags=max_lag, fft=True)
    g2m_potts_acf = acf(g2m_potts, nlags=max_lag, fft=True)

    # Potts Energy Autocorrelation: Barplot with envelope
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=200)
    axs[0].bar(np.arange(len(g1_potts_acf)), g1_potts_acf, color="#5E81AC", alpha=0.6, label='Autocorr')
    axs[0].plot(np.arange(len(g1_potts_acf)), g1_potts_acf, color="#5E81AC", linewidth=2, label='Envelope')
    axs[0].set_title('Potts Energy Autocorrelation (G1 phase)', fontsize=14)
    axs[0].set_ylabel('Autocorrelation', fontsize=12)
    axs[0].grid(True)

    axs[1].bar(np.arange(len(s_potts_acf)), s_potts_acf, color="#A3BE8C", alpha=0.6, label='Autocorr')
    axs[1].plot(np.arange(len(s_potts_acf)), s_potts_acf, color="#A3BE8C", linewidth=2, label='Envelope')
    axs[1].set_title('Potts Energy Autocorrelation (S phase)', fontsize=14)
    axs[1].set_ylabel('Autocorrelation', fontsize=12)
    axs[1].grid(True)

    axs[2].bar(np.arange(len(g2m_potts_acf)), g2m_potts_acf, color="#D08770", alpha=0.6, label='Autocorr')
    axs[2].plot(np.arange(len(g2m_potts_acf)), g2m_potts_acf, color="#D08770", linewidth=2, label='Envelope')
    axs[2].set_title('Potts Energy Autocorrelation (G2M phase)', fontsize=14)
    axs[2].set_xlabel('Lag', fontsize=12)
    axs[2].set_ylabel('Autocorrelation', fontsize=12)
    axs[2].grid(True)
    
    plt.tight_layout()
    if path:
        plt.savefig(path + '/plots/MCMC_diagnostics/potts_energy_autocorr.png', format='png', dpi=200)
        plt.savefig(path + '/plots/MCMC_diagnostics/potts_energy_autocorr.svg', format='svg', dpi=200)
    plt.close()

def ising_traj_plot(spins, save_path):
    plt.figure(figsize=(10, 10), dpi=200)
    # Use a diverging colormap that covers -2 (blue), 0 (white), 2 (red) with visible intermediate colors
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    plt.imshow(spins, cmap='Spectral_r', norm=norm, aspect='auto')
    plt.xlabel('MC step', fontsize=16)
    plt.ylabel('Simulation Beads', fontsize=16)
    plt.colorbar(label='Spin State')
    plt.savefig(save_path + '/plots/MCMC_diagnostics/potts_traj.png', format='png', dpi=200)
    plt.savefig(save_path + '/plots/MCMC_diagnostics/potts_traj.svg', format='svg', dpi=200)
    plt.close()