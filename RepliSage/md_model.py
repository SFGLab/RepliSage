#########################################################################
########### CREATOR: SEBASTIAN KORSAK, WARSAW 2022 ######################
#########################################################################

import numpy as np
import time
import openmm as mm
from tqdm import tqdm
from openmm.mtsintegrator import MTSLangevinIntegrator
from openmm.app import PDBxFile, ForceField, Simulation
from initial_structures import *
from utils import *

class MD_MODEL:
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
        self.rw_l = np.sqrt(self.N_beads)*0.1
        self.run_repli = np.all(rep_frac!=None)
        self.chain_idx = list()
        print('Average random walk distance:',self.rw_l)
        for i in range(N_beads): self.chain_idx.append(0)
        for i in range(N_beads,2*N_beads): self.chain_idx.append(1)

    def compute_rep(self,i):
        if i*self.step<self.t_rep:
            rep_per = 0
        elif i*self.step>=self.t_rep and i*self.step<self.t_rep+self.rep_duration:
            rep_per = np.count_nonzero(self.replicated_dna[:,i*self.step-self.t_rep])/self.N_beads*100
        else:
            rep_per = 100
        return rep_per

    def add_label(self,i):
        if i*self.step<self.t_rep:
            return 'BR', i-self.burnin+1
        elif i*self.step>=self.t_rep and i*self.step<self.t_rep+self.rep_duration:
            return 'R', i-self.t_rep//self.step+1
        else:
            return 'AR', i-(self.t_rep+self.rep_duration)//self.step+1
        return rep_per

    def run_pipeline(self,init_struct='hilbert', tol=1.0, sim_step=100, reporters=False,mode='MD',integrator_mode='langevin'):
        '''
        This is the basic function that runs the molecular simulation pipeline.

        Input parameters:
        run_MD (bool): True if user wants to run molecular simulation (not only energy minimization).
        sim_step (int): the simulation step of Langevin integrator.
        write_files (bool): True if the user wants to save the structures that determine the simulation ensemble.
        '''

        # Define initial structure
        print('\nBuilding initial structure...')
        points1 = compute_init_struct(self.N_beads,init_struct)
        points2 = points1 + [0.2,0.2,0.2] if self.run_repli else None
        write_mmcif(points1,points2,self.out_path+'/LE_init_struct.cif')
        generate_psf(self.N_beads,self.out_path+'/other/LE_init_struct.psf')
        print('Initial Structure Created Succesfully <3\n')
        
        platform = mm.Platform.getPlatformByName(self.platform)

        # Set up simulation
        pdb = PDBxFile(self.out_path+'/LE_init_struct.cif')
        forcefield = ForceField('forcefields/classic_sm_ff.xml')

        # Define the system
        self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=2*self.rw_l)
        if integrator_mode=='langevin':
            integrator = mm.LangevinIntegrator(310*mm.unit.kelvin, 0.1/mm.unit.femtosecond, 10 * mm.unit.femtosecond)
        elif integrator_mode=='brownian':
            integrator = mm.BrownianIntegrator(310*mm.unit.kelvin, 0.1/mm.unit.femtosecond, 10 * mm.unit.femtosecond)
        
        # Forcefield and Simulation Definition
        self.add_forcefield(self.burnin)
        self.simulation = Simulation(pdb.topology, self.system, integrator, platform)
        self.simulation.context.setPositions(pdb.positions)

        # Run energy minimization
        print('\nRunning initial energy minimization...')
        start = time.time()
        self.simulation.minimizeEnergy(tolerance=tol)
        self.state = self.simulation.context.getState(getPositions=True)
        if reporters:
            self.simulation.reporters.append(StateDataReporter(stdout, (self.N_steps*sim_step)//1000, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
            self.simulation.reporters.append(DCDReporter(self.path+'/other/replisage.dcd', sim_step))
        PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.out_path+f'/minimized_model.cif', 'w'))
        end = time.time()
        elapsed = end - start
        print(f'Energy minimization finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')
        
        print(f'\nCreating ensembles ({mode} mode)...')
        start = time.time()
        if not reporters: pbar = tqdm(total=self.N_steps-self.burnin, desc='Progress of Simulation.')
        for i in range(self.burnin,self.N_steps):
            # Compute replicated DNA
            rep_per = self.compute_rep(i)
            
            # Change forces
            self.change_repliforce(i)
            self.change_loop(i)
            if p_ev>0: self.ps_ev = np.random.rand(self.N_beads)
            self.change_ev()
            if self.Cs.ndim>1: self.change_comps(i)

            # Update steps of simulations
            if mode=='MD':
                self.simulation.step(sim_step)
            elif mode=='EM':
                self.simulation.minimizeEnergy(tolerance=tol)
            else:
                raise InterruptedError('Mode can be only EM or MD.')
            label, idx = self.add_label(i)

            # Save structure
            self.state = self.simulation.context.getState(getPositions=True)
            PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.out_path+f'/ensemble/ensemble_{i-self.burnin+1}_{label}.cif', 'w'))
            
            # Update progress-bar
            if not reporters:
                pbar.update(1)
                pbar.set_description(f'Percentage of replicated dna {rep_per:.1f}%')
        if not reporters: pbar.close()
        end = time.time()
        elapsed = end - start
        print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')
        print('Energy minimization done <3')

    def change_ev(self):
        ev_strength = (self.ps_ev>self.p_ev).astype(int)*10 if self.p_ev>0 else 10*np.ones(self.N_beads)
        for n in range(self.N_beads):
            self.ev_force.setParticleParameters(n,[ev_strength[n],0.05])
        self.ev_force.updateParametersInContext(self.simulation.context)

    def change_repliforce(self,i):
        if i*self.step>=self.t_rep and i*self.step<self.t_rep+self.rep_duration:
            rep_dna = self.replicated_dna[:,i*self.step-self.t_rep]
            rep_locs = np.nonzero(rep_dna)[0]
            for l in rep_locs:
                self.repli_force.setBondParameters(int(l),int(l),int(l)+self.N_beads,[2*self.rw_l,1e3])
        elif i*self.step>=self.t_rep+self.rep_duration:
            for j in range(self.N_beads):
                self.repli_force.setBondParameters(j,j,j+self.N_beads,[4*self.rw_l,1e3])
        self.repli_force.updateParametersInContext(self.simulation.context)

    def change_loop(self,i):
        force_idx = self.system.getNumForces()-1
        self.system.removeForce(force_idx)
        self.add_loops(i)
        self.simulation.context.reinitialize(preserveState=True)
        self.LE_force.updateParametersInContext(self.simulation.context)

    def change_comps(self,i):
        for n in range(self.N_beads):
            self.comp_force.setParticleParameters(n,[self.Cs[n,i],self.chain_idx[n]])
        if self.run_repli:
            for n in range(self.N_beads,2*self.N_beads):
                self.comp_force.setParticleParameters(n,[self.Cs[n%self.N_beads,i],self.chain_idx[n]])
        self.comp_force.updateParametersInContext(self.simulation.context)

    def add_evforce(self):
        'Leonard-Jones potential for excluded volume'
        self.ev_force = mm.CustomNonbondedForce(f'(epsilon1*epsilon2*(sigma1*sigma2)/(r+r_small))^3')
        self.ev_force.addGlobalParameter('r_small', defaultValue=0.1)
        self.ev_force.addPerParticleParameter('sigma')
        self.ev_force.addPerParticleParameter('epsilon')
        self.ev_force.setCutoffDistance(distance=0.2)
        self.ev_force.setForceGroup(1)
        for i in range(self.N_beads):
            self.ev_force.addParticle([10,0.05])
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads):
                self.ev_force.addParticle([10,0.05])
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
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 500)
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads - 2):
                self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 500)
        self.system.addForce(self.angle_force)
    
    def add_loops(self,i):
        'LE force that connects cohesin restraints'
        self.LE_force = mm.HarmonicBondForce()
        self.LE_force.setForceGroup(0)
        for n in range(self.N_coh):
            self.LE_force.addBond(self.M[n,i], self.N[n,i], 0.1, 5e4)
            if self.run_repli: 
                self.LE_force.addBond(self.N_beads+self.M[n,i], self.N_beads+self.N[n,i], 0.1, 5e4)
        self.system.addForce(self.LE_force)
    
    def add_repliforce(self,i):
        'Replication force to bring together the two polymers'
        self.repli_force = mm.CustomBondForce('D * (r-r0)^2')
        self.repli_force.setForceGroup(0)
        self.repli_force.addPerBondParameter('r0')
        self.repli_force.addPerBondParameter('D')
        
        if i*self.step<self.t_rep:
            for i in range(self.N_beads): 
                self.repli_force.addBond(i, i + self.N_beads, [0.1,5e5])
        elif i*self.step>=self.t_rep and i*self.step<self.t_rep+self.rep_duration:
            rep_dna = self.replicated_dna[:,i*self.step-self.t_rep]
            rep_locs = np.nonzero(rep_dna)[0]
            for i in range(self.N_beads):
                if i in rep_locs:
                    self.repli_force.addBond(i, i + self.N_beads, [2*self.rw_l,1e3])
                else:
                    self.repli_force.addBond(i, i + self.N_beads, [0.1,5e5])
        else:
            for i in range(self.N_beads):
                self.repli_force.addBond(i, i + self.N_beads, [4*self.rw_l,1e3])
        
        self.system.addForce(self.repli_force)
    
    def add_blocks(self,i):
        'Block copolymer forcefield for the modelling of compartments.'
        cs = self.Cs[:,i] if self.Cs.ndim>1 else self.Cs
        self.comp_force = mm.CustomNonbondedForce('delta(c1-c2)*E*exp(-(r-r0)^2/(2*sigma^2)); E=Ea1*delta(s1-2)*delta(s2-2)+Ea2*delta(s1-1)*delta(s2-1)+Eb1*delta(s1)*delta(s2)+Eb2*delta(s1+1)*delta(s2+1)+Eb3*delta(s1+2)*delta(s2+2)')
        self.comp_force.setForceGroup(1)
        self.comp_force.addGlobalParameter('sigma',defaultValue=self.rw_l/2)
        self.comp_force.addGlobalParameter('r0',defaultValue=0.2)
        self.comp_force.addGlobalParameter('Ea1',defaultValue=-0.1)
        self.comp_force.addGlobalParameter('Ea2',defaultValue=-0.2)
        self.comp_force.addGlobalParameter('Eb1',defaultValue=-0.3)
        self.comp_force.addGlobalParameter('Eb2',defaultValue=-0.4)
        self.comp_force.addGlobalParameter('Eb3',defaultValue=-0.5)
        # self.comp_force.setCutoffDistance(distance=self.rw_l)
        self.comp_force.addPerParticleParameter('s')
        self.comp_force.addPerParticleParameter('c')
        for i in range(self.N_beads):
            self.comp_force.addParticle([cs[i],self.chain_idx[i]])
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads):
                self.comp_force.addParticle([cs[i%self.N_beads],self.chain_idx[i]])
        self.system.addForce(self.comp_force)
    
    def add_container(self, R=10.0, C=1.0):
        self.container_force = mm.CustomNonbondedForce('C*(max(0, r-R)^2)*delta(c1-c2)')
        self.container_force.setForceGroup(1)
        self.container_force.addGlobalParameter('C',defaultValue=C)
        self.container_force.addGlobalParameter('R',defaultValue=R)
        self.container_force.setCutoffDistance(2*self.rw_l)
        self.container_force.addPerParticleParameter('c')
        for i in range(self.N_beads):
            self.container_force.addParticle([self.chain_idx[i]])
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads):
                self.container_force.addParticle([self.chain_idx[i]])
        self.system.addForce(self.container_force)
    
    def add_forcefield(self,i,use_container=True):
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
        if use_container: self.add_container(R = self.rw_l)
        if np.all(self.Cs!=None): self.add_blocks(i)
        if self.run_repli: self.add_repliforce(i)
        self.add_loops(i)