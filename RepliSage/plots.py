import shutil
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from tqdm import tqdm
import time

def make_loop_hist(Ms,Ns,path=None):
    Ls = np.abs(Ns-Ms).flatten()
    Ls_df = pd.DataFrame(Ls)
    figure(figsize=(10, 7), dpi=200)
    sns.histplot(data=Ls_df, bins=30,  kde=True,stat='density')
    plt.grid()
    plt.legend()
    plt.ylabel('Probability',fontsize=16)
    plt.xlabel('Loop Length',fontsize=16)
    if path!=None:
        save_path = path+'/plots/loop_length.png'
        plt.savefig(save_path,format='png',dpi=200)
        save_path = path+'/plots/loop_length.svg'
        plt.savefig(save_path,format='svg',dpi=200)
        save_path = path+'/plots/loop_length.pdf'
        plt.savefig(save_path,format='pdf',dpi=200)
    plt.close()

    Is, Js = Ms.flatten(), Ns.flatten()
    IJ_df = pd.DataFrame()
    IJ_df['mi'] = Is
    IJ_df['nj'] = Js
    figure(figsize=(8, 8), dpi=200)
    sns.jointplot(IJ_df, x="mi", y="nj",kind='hex',color='Red')
    if path!=None:
        save_path = path+'/plots/ij_prob.png'
        plt.savefig(save_path,format='png',dpi=200)
        save_path = path+'/plots/ij_prob.svg'
        plt.savefig(save_path,format='svg',dpi=200)
        save_path = path+'/plots/ij_prob.pdf'
        plt.savefig(save_path,format='pdf',dpi=200)
    plt.close()

def make_moveplots(unbinds, slides, path=None):
    figure(figsize=(10, 8), dpi=200)
    plt.plot(unbinds, 'blue')
    plt.plot(slides, 'red')
    plt.ylabel('Number of moves', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(['Rebinding', 'Sliding'], fontsize=16)
    plt.grid()
    if path!=None:
        save_path = path+'/plots/moveplot.png'
        plt.savefig(save_path,dpi=200)
        save_path = path+'/plots/moveplot.pdf'
        plt.savefig(save_path,dpi=200)
    plt.close()

def coh_traj_plot(ms,ns,N_beads,path):
    print('\nPlotting trajectories of cohesins...')
    start = time.time()
    N_coh = len(ms)
    figure(figsize=(20, 15),dpi=200)
    color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) for i in range(N_coh)]
    size = 0.1
    
    for nn in tqdm(range(N_coh)):
        tr_m, tr_n = ms[nn], ns[nn]
        plt.fill_between(np.arange(len(tr_m)), tr_m, tr_n, color=color[nn], alpha=0.3, interpolate=False, linewidth=0)
    plt.xlabel('Simulation Step', fontsize=16)
    plt.ylabel('Position of Cohesin', fontsize=16)
    plt.gca().invert_yaxis()
    save_path = path+'/plots/LEFs.png'
    plt.savefig(save_path,format='png')
    plt.show()
    end = time.time()
    elapsed = end - start
    print(f'Plot created succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

def make_timeplots(Es, Fs, Bs, Rs, burnin, path=None):
    plt.plot(Es, 'k',label='Total Energy')
    plt.plot(Fs, 'b',label='Folding Energy')
    plt.plot(Bs, 'r',label='Binding Energy')
    plt.plot(Rs, 'g',label='Replication Energy')
    plt.ylabel('Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    plt.yscale('symlog')
    plt.legend()
    save_path = path+'/plots/total_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/total_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    plt.show()

    plt.plot(Fs, 'b')
    plt.ylabel('Folding Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/fold_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/fold_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    plt.show()

    plt.plot(Rs, 'g')
    plt.ylabel('Replication Energy', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    save_path = path+'/plots/repli_energy.pdf'
    plt.savefig(save_path,format='pdf',dpi=200)
    save_path = path+'/plots/repli_energy.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    plt.show()

    ys = np.array(Fs)[burnin:]
    plot_acf(ys, title=None, lags = len(np.array(Fs)[burnin:])//2)
    plt.ylabel("Autocorrelations", fontsize=16)
    plt.xlabel("Lags", fontsize=16)
    plt.grid()
    if path!=None: 
        save_path = path+'/plots/autoc.png'
        plt.savefig(save_path,dpi=200)
        save_path = path+'/plots/autoc.svg'
        plt.savefig(save_path,format='svg',dpi=200)
        save_path = path+'/plots/autoc.pdf'
        plt.savefig(save_path,format='pdf',dpi=200)
    plt.show()

def ising_traj_plot(traj,save_path):
    figure(figsize=(10, 15),dpi=100)
    plt.imshow(traj,cmap='bwr_r',aspect='auto')
    plt.xlabel('Computational Time',fontsize=20)
    plt.ylabel('Region', fontsize=20)
    plt.title('Epigenetic Mark Spread',fontsize=25)
    plt.savefig(save_path+'/plots/ising_traj.png',format='png',dpi=100)
    plt.show()