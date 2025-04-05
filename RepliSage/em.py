#########################################################################
########### CREATOR: SEBASTIAN KORSAK, WARSAW 2022 ######################
#########################################################################

import numpy as np
import time
import openmm as mm
from tqdm import tqdm
from openmm.mtsintegrator import MTSLangevinIntegrator
from openmm.app import PDBxFile, ForceField, Simulation
from .initial_structures import *
from .utils import *

class EM_LE:
    def __init__(self,M,N,N_beads,burnin,MC_step,out_path,platform,rep_frac=None,t_rep=None,Cs=None):
        '''
        M, N (np arrays): Position matrix of two legs of cohesin m,n. 
                          Rows represent  loops/cohesins and columns represent time
        N_beads (int): The number of beads of initial structure.
        step (int): sampling rate
        out_path (int): the out_path where the simulation will save structures etc.
        '''
        self.M, self.N, self.Cs = M, N, Cs
        self.replicated_dna, self.t_rep = rep_frac, t_rep
        self.rep_duration = len(self.replicated_dna[0,:])
        self.N_coh, self.N_steps = M.shape
        self.N_beads, self.step, self.burnin = N_beads, MC_step, burnin//MC_step
        self.out_path = out_path
        self.platform = platform
        self.run_repli = np.all(rep_frac!=None)

    def run_pipeline(self,init_struct='hilbert',tol=1.0):
        '''
        This is the basic function that runs the molecular simulation pipeline.

        Input parameters:
        run_MD (bool): True if user wants to run molecular simulation (not only energy minimization).
        sim_step (int): the simulation step of Langevin integrator.
        write_files (bool): True if the user wants to save the structures that determine the simulation ensemble.
        '''
        # Define initial structure
        print('Building initial structure...')
        points1 = compute_init_struct(self.N_beads,init_struct)
        points2 = points1 + [0.2,0.2,0.2] if self.run_repli else None
        write_mmcif(points1,points2,self.out_path+'/LE_init_struct.cif')
        generate_psf(self.N_beads,self.out_path+'/other/LE_init_struct.psf')
        print('Done brother ;D\n')
        
        # Define System
        print('Minimizing energy...')
        start = time.time()
        pbar = tqdm(total=self.N_steps-self.burnin, desc='Progress of Simulation.')
        platform = mm.Platform.getPlatformByName(self.platform)
        for i in range(self.burnin,self.N_steps):
            # Compyte replicated DNA
            if i*self.step<self.t_rep:
                rep_per = 0
            elif i*self.step>=self.t_rep and i*self.step<self.t_rep+self.rep_duration:
                rep_per = np.count_nonzero(self.replicated_dna[:,i*self.step-self.t_rep])/self.N_beads*100
            else:
                rep_per = 100

            # Set up simulation
            pdb = PDBxFile(self.out_path+'/LE_init_struct.cif')
            forcefield = ForceField('forcefields/classic_sm_ff.xml')
            self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=2*np.sqrt(self.N_beads)*0.1)
            integrator = MTSLangevinIntegrator(300*mm.unit.kelvin, 0.1/mm.unit.picosecond, 10 * mm.unit.femtosecond, [(0,4),(1,2),(0,1)])
            
            # Forcefield
            ms,ns=self.M[:,i], self.N[:,i]
            cs = self.Cs[:,i] if self.Cs.ndim>1 else self.Cs
            self.add_forcefield(ms,ns,cs)
            
            # Minimize energy
            self.simulation = Simulation(pdb.topology, self.system, integrator, platform)
            if self.run_repli: self.change_repliforce(i)
            self.simulation.context.setPositions(pdb.positions)
            self.simulation.minimizeEnergy(tolerance=tol)
            self.state = self.simulation.context.getState(getPositions=True)
            PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.out_path+f'/ensemble/model_{i-self.burnin+1}.cif', 'w'))
            
            # Update progress-bar
            pbar.update(1)
            pbar.set_description(f'Percentage of replicated dna {rep_per:.1f}%')
        pbar.close()
        end = time.time()
        elapsed = end - start
        print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')
        print('Energy minimization done :D')

    def change_repliforce(self,i):
        if i*self.step>=self.t_rep and i*self.step<self.t_rep+self.rep_duration:
            rep_dna = self.replicated_dna[:,i*self.step-self.t_rep]
            rep_locs = np.nonzero(rep_dna)[0]
            for l in rep_locs:
                self.repli_force.setBondParameters(int(l),int(l),int(l)+self.N_beads,[0.5*np.sqrt(self.N_beads)*0.1,1e2])
        elif i*self.step>=self.t_rep+self.rep_duration:
            for j in range(self.N_beads):
                self.repli_force.setBondParameters(j,j,j+self.N_beads,[2*np.sqrt(self.N_beads)*0.1,1.0])
        self.repli_force.updateParametersInContext(self.simulation.context)

    def add_evforce(self):
        'Leonard-Jones potential for excluded volume'
        self.ev_force = mm.CustomNonbondedForce('epsilon*(r_ev/r+delta)^3')
        self.ev_force.addGlobalParameter('epsilon', defaultValue=10)
        self.ev_force.addGlobalParameter('r_ev',defaultValue=0.1)
        self.ev_force.addGlobalParameter('delta',defaultValue=0.001)
        self.ev_force.setCutoffDistance(distance=0.2)
        self.ev_force.setForceGroup(0)
        for i in range(self.N_beads):
            self.ev_force.addParticle()
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads):
                self.ev_force.addParticle()
            for i in range(self.N_beads):
                for j in range(self.N_beads,2*self.N_beads):
                    self.ev_force.addExclusion(i,j)
        self.system.addForce(self.ev_force)

    def add_bonds(self):
        'Harmonic bond borce between succesive beads'
        self.bond_force = mm.HarmonicBondForce()
        self.bond_force.setForceGroup(0)
        for i in range(self.N_beads - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 3e5)
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads - 1):
                self.bond_force.addBond(i, i + 1, 0.1, 3e5)
        self.system.addForce(self.bond_force)
    
    def add_stiffness(self):
        'Harmonic angle force between successive beads so as to make chromatin rigid'
        self.angle_force = mm.HarmonicAngleForce()
        self.angle_force.setForceGroup(0)
        for i in range(self.N_beads - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 10)
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads - 2):
                self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 10)
        self.system.addForce(self.angle_force)
    
    def add_loops(self,ms,ns,i=0):
        'LE force that connects cohesin restraints'
        self.LE_force = mm.HarmonicBondForce()
        self.LE_force.setForceGroup(0)
        for i in range(self.N_coh):
            self.LE_force.addBond(ms[i], ns[i], 0.1, 5e4)
            if self.run_repli: self.LE_force.addBond(self.N_beads+ms[i], self.N_beads+ns[i], 0.1, 5e4)
        self.system.addForce(self.LE_force)
    
    def add_repliforce(self):
        'Replication force to bring together the two polymers'
        self.repli_force = mm.CustomBondForce('D * (r-r0)^2')
        self.repli_force.setForceGroup(0)
        self.repli_force.addPerBondParameter('r0')
        self.repli_force.addPerBondParameter('D')
        for i in range(self.N_beads):
            self.repli_force.addBond(i, i + self.N_beads, [0,5e4])
        self.system.addForce(self.repli_force)
    
    def add_blocks(self,cs):
        'Block copolymer forcefield for the modelling of compartments.'
        self.comp_force = mm.CustomNonbondedForce('E*exp(-(r-r0)^2/(2*sigma^2)); E=Ea*delta(s1-1)*delta(s2-1)+Eb*delta(s1+1)*delta(s2+1)')
        self.comp_force.setForceGroup(1)
        self.comp_force.addGlobalParameter('sigma',defaultValue=0.4)
        self.comp_force.addGlobalParameter('r0',defaultValue=0.2)
        self.comp_force.addGlobalParameter('Ea',defaultValue=-0.5)
        self.comp_force.addGlobalParameter('Eb',defaultValue=-1.0)
        self.comp_force.addPerParticleParameter('s')
        self.comp_force.setCutoffDistance(distance=2.0)
        for i in range(self.N_beads):
            self.comp_force.addParticle([cs[i]])
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads):
                self.comp_force.addParticle([cs[i%self.N_beads]])
            for i in range(self.N_beads):
                for j in range(self.N_beads,2*self.N_beads):
                    self.comp_force.addExclusion(i,j)
        self.system.addForce(self.comp_force)

    def add_container(self, R=10.0, C=10.0):
        self.container_force = mm.CustomNonbondedForce('C*(max(0, r-R)^2)')
        self.container_force.setForceGroup(1)
        self.container_force.addGlobalParameter('C',defaultValue=C)
        self.container_force.addGlobalParameter('R',defaultValue=R)
        self.container_force.setCutoffDistance(2*np.sqrt(self.N_beads)*0.1)
        for i in range(self.N_beads):
            self.container_force.addParticle()
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads):
                self.container_force.addParticle()
            for i in range(self.N_beads):
                for j in range(self.N_beads,2*self.N_beads):
                    self.container_force.addExclusion(i,j)
        self.system.addForce(self.container_force)
    
    def add_forcefield(self,ms,ns,cs=None,use_container=True):
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
        if use_container: self.add_container(R = 1.5*np.sqrt(self.N_beads)*0.1)
        if np.all(cs!=None): self.add_blocks(cs)
        if self.run_repli: self.add_repliforce()
        self.add_loops(ms,ns)