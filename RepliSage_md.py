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

class MD_LE:
    def __init__(self,M,N,l_forks,r_forks,t_rep,N_beads,burnin,MC_step,out_path,platform,Cs=None):
        '''
        M, N (np arrays): Position matrix of two legs of cohesin m,n. 
                          Rows represent  loops/cohesins and columns represent time
        N_beads (int): The number of beads of initial structure.
        step (int): sampling rate
        out_path (int): the out_path where the simulation will save structures etc.
        '''
        self.M, self.N, self.Cs = M, N, Cs
        self.l_forks, self.r_forks, self.t_rep = l_forks, r_forks, t_rep
        self.rep_duration = len(self.l_forks[0,:])
        print('Duration of Replication',self.rep_duration)
        self.N_coh, self.N_steps = M.shape
        self.N_beads, self.step, self.burnin = N_beads, MC_step, burnin//MC_step
        self.out_path = out_path
        self.platform = platform
    
    def run_pipeline(self,run_MD=True,sim_step=10,write_files=False):
        '''
        This is the basic function that runs the molecular simulation pipeline.

        Input parameters:
        run_MD (bool): True if user wants to run molecular simulation (not only energy minimization).
        sim_step (int): the simulation step of Langevin integrator.
        write_files (bool): True if the user wants to save the structures that determine the simulation ensemble.
        '''
        # Define initial structure
        print('Building initial structure...')
        points1 = polymer_circle(self.N_beads)
        points2 = points1 + [0.2,0.2,0.2]
        write_mmcif(points1,points2,self.out_path+'/LE_init_struct.cif')
        generate_psf(self.N_beads,self.out_path+'/other/LE_init_struct.psf')
        print('Done brother ;D\n')
        
        # Define System
        pdb = PDBxFile(self.out_path+'/LE_init_struct.cif')
        forcefield = ForceField('forcefields/classic_sm_ff.xml')
        self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1*u.nanometer)
        integrator = mm.LangevinIntegrator(310, 0.1, 50 * mm.unit.femtosecond)

        # Add forces
        print('Adding forces...')
        self.add_forcefield()
        print('Forces added ;)\n')
        
        # Minimize energy
        print('Minimizing energy...')
        platform = mm.Platform.getPlatformByName(self.platform)
        self.simulation = Simulation(pdb.topology, self.system, integrator, platform)
        self.simulation.reporters.append(StateDataReporter(stdout, (self.N_steps*sim_step)//100, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
        self.simulation.reporters.append(DCDReporter(self.out_path+'/other/stochastic_LE.dcd', 100))
        self.simulation.context.setPositions(pdb.positions)
        current_platform = self.simulation.context.getPlatform()
        print(f"Simulation will run on platform: {current_platform.getName()}")
        self.simulation.minimizeEnergy()
        print('Energy minimization done :D\n')

        # Run molecular dynamics self.simulation
        if run_MD:
            print('Running molecular dynamics...')
            start = time.time()
            for i in range(1,self.N_steps):
                if np.all(self.Cs!=None): self.change_blocks(i)
                self.change_loop(i)
                if i>=self.t_rep: self.change_repliforce(i)
                self.simulation.step(sim_step)
                if i%self.step==0 and i>self.burnin*self.step:
                    self.state = self.simulation.context.getState(getPositions=True)
                    if write_files: PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.out_path+f'/pdbs/MDLE_{i//self.step-self.burnin}.cif', 'w'))
                    time.sleep(2)
            end = time.time()
            elapsed = end - start

            print(f'Everything is done! simulation finished succesfully!\nMD finished in {elapsed/60:.2f} minutes.\n')

    def change_blocks(self,i):
        for j in range(2*self.N_beads):
            self.comp_force.setParticleParameters(j,[self.Cs[j%self.N_beads,i]])
        self.comp_force.updateParametersInContext(self.simulation.context)

    def change_loop(self,i):
        force_idx = self.system.getNumForces()-1
        self.system.removeForce(force_idx)
        self.add_loops(i)
        self.simulation.context.reinitialize(preserveState=True)
        self.LE_force.updateParametersInContext(self.simulation.context)

    def change_repliforce(self,i):
        if i>=self.t_rep and i<self.t_rep+self.rep_duration:
            locs1 = np.nonzero(self.l_forks[:,i-self.t_rep])[0]
            locs2 = np.nonzero(self.r_forks[:,i-self.t_rep])[0]
            locs = np.union1d(locs1,locs2)
            for l in locs:
                self.repli_force.setBondParameters(int(l),int(l),int(l)+self.N_beads,[0.4,5e1])
        elif i>=self.t_rep+self.rep_duration:
            for j in range(self.N_beads):
                self.repli_force.setBondParameters(j,j,j+self.N_beads,[10.0,5.0])
        self.repli_force.updateParametersInContext(self.simulation.context)

    def add_evforce(self):
        'Leonard-Jones potential for excluded volume'
        self.ev_force = mm.CustomNonbondedForce('epsilon*((sigma1+sigma2)/(r+r_small))^3')
        self.ev_force.addGlobalParameter('epsilon', defaultValue=5)
        self.ev_force.addGlobalParameter('r_small', defaultValue=0.01)
        self.ev_force.addPerParticleParameter('sigma')
        for i in range(2*self.N_beads):
            self.ev_force.addParticle([0.05])
        for i in range(self.N_beads):
            self.ev_force.addExclusion(i,i+self.N_beads)
        self.system.addForce(self.ev_force)

    def add_bonds(self):
        'Harmonic bond borce between succesive beads'
        self.bond_force = mm.HarmonicBondForce()
        for i in range(self.N_beads - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 3e5)
        for i in range(self.N_beads,2*self.N_beads - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 3e5)
        self.system.addForce(self.bond_force)
    
    def add_stiffness(self):
        'Harmonic angle force between successive beads so as to make chromatin rigid'
        self.angle_force = mm.HarmonicAngleForce()
        for i in range(self.N_beads - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 200)
        for i in range(self.N_beads,2*self.N_beads - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 200)
        self.system.addForce(self.angle_force)
    
    def add_loops(self,i=0):
        'LE force that connects cohesin restraints'
        self.LE_force = mm.HarmonicBondForce()
        for nn in range(self.N_coh):
            self.LE_force.addBond(self.M[nn,i], self.N[nn,i], 0.05, 3e4)
            self.LE_force.addBond(self.N_beads+self.M[nn,i], self.N_beads+self.N[nn,i], 0.05, 3e4)
        self.system.addForce(self.LE_force)
    
    def add_repliforce(self):
        'Replication force to bring together the two polymers'
        self.repli_force = mm.CustomBondForce('D * (r-r0)^2')
        self.repli_force.addPerBondParameter('r0')
        self.repli_force.addPerBondParameter('D')
        for i in range(self.N_beads):
            self.repli_force.addBond(i, i + self.N_beads, [0,5e5])
        self.system.addForce(self.repli_force)

    def add_blocks(self):
        'Block copolymer forcefield for the modelling of compartments.'
        self.comp_force = mm.CustomNonbondedForce('E*exp(-(r-r0)^2/(2*sigma^2)); E=Ea*delta(s1-1)*delta(s2-1)+Eb*delta(s1+1)*delta(s2+1)')
        self.comp_force.addGlobalParameter('sigma',defaultValue=1)
        self.comp_force.addGlobalParameter('r0',defaultValue=0.4)
        self.comp_force.addGlobalParameter('Ea',defaultValue=-2.0)
        self.comp_force.addGlobalParameter('Eb',defaultValue=-4.0)
        self.comp_force.addPerParticleParameter('s')
        for i in range(2*self.N_beads):
            self.comp_force.addParticle([self.Cs[i%self.N_beads,0]])
        for i in range(self.N_beads):
            self.comp_force.addExclusion(i,i+self.N_beads)
        self.system.addForce(self.comp_force)
    
    def add_forcefield(self):
        '''
        Here is the definition of the forcefield.

        There are the following energies:
        - ev force: repelling LJ-like forcefield
        - harmonic bond force: to connect adjacent beads.
        - angle force: for polymer stiffness.
        - LE forces: this is a list of force objects. Each object corresponds to a different cohesin. It is needed to define a force for each time step.
        '''
        self.add_evforce()
        self.add_bonds()
        self.add_stiffness()
        if np.all(self.Cs!=None): self.add_blocks()
        self.add_repliforce()
        self.add_loops()
        
def main():
    # A potential example
    M = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ms.npy')
    N = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ns.npy')
    md = MD_LE(4*M,4*N,2000,5,1)
    md.run_pipeline(write_files=False,plots=True,sim_step=100)
