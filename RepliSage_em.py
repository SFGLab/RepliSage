#########################################################################
########### CREATOR: SEBASTIAN KORSAK, WARSAW 2022 ######################
#########################################################################

import copy
import time
import numpy as np
import openmm as mm
import openmm.unit as u
from tqdm import tqdm
from sys import stdout
from mdtraj.reporters import HDF5Reporter
from scipy import ndimage
from openmm.app import PDBFile, PDBxFile, ForceField, Simulation, PDBReporter, PDBxReporter, DCDReporter, StateDataReporter, CharmmPsfFile
from RepliSage_utils import *

class EM_LE:
    def __init__(self,M,N,N_beads,burnin,MC_step,path,platform):
        '''
        M, N (np arrays): Position matrix of two legs of cohesin m,n. 
                          Rows represent  loops/cohesins and columns represent time
        N_beads (int): The number of beads of initial structure.
        step (int): sampling rate
        path (int): the path where the simulation will save structures etc.
        '''
        self.M, self.N = M, N
        self.N_coh, self.N_steps = M.shape
        self.N_beads, self.step, self.burnin = N_beads, MC_step, burnin//MC_step
        self.path = path
        self.platform = platform
    
    def run_pipeline(self,crash_step=5,write_files=False,plots=False):
        '''
        This is the basic function that runs the molecular simulation pipeline.

        Input parameters:
        run_MD (bool): True if user wants to run molecular simulation (not only energy minimization).
        sim_step (int): the simulation step of Langevin integrator.
        write_files (bool): True if the user wants to save the structures that determine the simulation ensemble.
        plots (bool): True if the user wants to see the output average heatmaps.
        '''
        # Define initial structure
        print('Building initial structure...')
        points = self_avoiding_random_walk(self.N_beads, 2, 0.001)
        write_mmcif(points,self.path+'/LE_init_struct.cif')
        generate_psf(self.N_beads,self.path+'/other/LE_init_struct.psf')
        print('Done brother ;D\n')

        counter = 0
        sum_heat = np.zeros((self.N_beads,self.N_beads))
        print('Running energy minimizations...')
        for i in tqdm(range(self.burnin,self.N_steps,self.step)):
            # Define System
            pdb = PDBxFile(self.path+'/LE_init_struct.cif')
            forcefield = ForceField('forcefields/classic_sm_ff.xml')
            self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1*u.nanometer)
            integrator = mm.LangevinIntegrator(310, 0.05, 100 * mm.unit.femtosecond)

            # Add forces
            ms,ns = self.M[:,i], self.N[:,i]
            self.add_forcefield(ms,ns)

            # Minimize energy
            platform = mm.Platform.getPlatformByName(self.platform)
            simulation = Simulation(pdb.topology, self.system, integrator, platform)
            simulation.context.setPositions(pdb.positions)
            simulation.minimizeEnergy(tolerance=0.001)
            self.state = simulation.context.getState(getPositions=True)
            PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.path+f'/pdbs/EMLE_{(i-self.burnin)//self.step}.cif', 'w'))
            save_path = self.path+f'/heatmaps/heat_{(i-self.burnin)//self.step}.svg' if write_files else None
            sum_heat+=get_heatmap(self.state.getPositions(),save_path=save_path,save=write_files)
            if (i-self.burnin)//self.step%crash_step==0: time.sleep(10) # so as to not overload the system
            counter+=1
        print('Energy minimizations done :D\n')

        self.avg_heat = sum_heat/counter
        np.save(self.path+f'/other/avg_heatmap.npy',self.avg_heat)
        if plots:
            figure(figsize=(10, 10))
            plt.imshow(self.avg_heat,cmap="Reds",vmax=1)
            plt.colorbar()
            plt.savefig(self.path+f'/plots/avg_heatmap.svg',format='svg',dpi=500)
            plt.savefig(self.path+f'/plots/avg_heatmap.pdf',format='pdf',dpi=500)
            # plt.colorbar()
            plt.close()
            
            return self.avg_heat

    def add_forcefield(self,ms,ns):
        '''
        Here is the definition of the forcefield.

        There are the following energies:
        - ev force: repelling LJ-like forcefield
        - harmonic bond force: to connect adjacent beads.
        - angle force: for polymer stiffness.
        - LE forces: this is a list of force objects. Each object corresponds to a different cohesin. It is needed to define a force for each time step.
        '''
        # Leonard-Jones potential for excluded volume
        self.ev_force = mm.CustomNonbondedForce('epsilon*((sigma1+sigma2)/r)^12')
        self.ev_force.addGlobalParameter('epsilon', defaultValue=10)
        self.ev_force.addPerParticleParameter('sigma')
        self.system.addForce(self.ev_force)
        for i in range(self.system.getNumParticles()):
            self.ev_force.addParticle([0.1])

        # Harmonic bond borce between succesive beads
        self.bond_force = mm.HarmonicBondForce()
        self.system.addForce(self.bond_force)
        for i in range(self.system.getNumParticles() - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 300000.0)

        # Harmonic angle force between successive beads so as to make chromatin rigid
        self.angle_force = mm.HarmonicAngleForce()
        self.system.addForce(self.angle_force)
        for i in range(self.system.getNumParticles() - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 200)
        
        # LE force that connects cohesin restraints
        self.LE_force = mm.HarmonicBondForce()
        self.system.addForce(self.LE_force)
        for nn in range(self.N_coh):
            self.LE_force.addBond(ms[nn], ns[nn], 0.1, 300000.0)
        
def main():
    # A potential example
    M = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ms.npy')
    N = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ns.npy')
    md = EM_LE(4*M,4*N,2000,5,1)
    md.run_pipeline(write_files=False,plots=True,sim_step=100)
