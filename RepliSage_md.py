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
    def __init__(self,M,N,l_forks,r_forks,t_rep,N_beads,burnin,MC_step,path,platform):
        '''
        M, N (np arrays): Position matrix of two legs of cohesin m,n. 
                          Rows represent  loops/cohesins and columns represent time
        N_beads (int): The number of beads of initial structure.
        step (int): sampling rate
        path (int): the path where the simulation will save structures etc.
        '''
        self.M, self.N = M, N
        self.l_forks, self.r_forks, self.t_rep = l_forks, r_forks, t_rep
        self.rep_duration = len(self.l_forks[0,:])
        self.N_coh, self.N_steps = M.shape
        self.N_beads, self.step, self.burnin = N_beads, MC_step, burnin//MC_step
        self.path = path
        self.platform = platform
    
    def run_pipeline(self,run_MD=True,sim_step=5,write_files=False,plots=False):
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
        points1 = polymer_circle(self.N_beads)
        points2 = points1 + [0.2,0.2,0.2]
        write_mmcif(points1,points2,self.path+'/LE_init_struct.cif')
        generate_psf(self.N_beads,self.path+'/other/LE_init_struct.psf')
        print('Done brother ;D\n')

        # Define System
        pdb = PDBxFile(self.path+'/LE_init_struct.cif')
        forcefield = ForceField('forcefields/classic_sm_ff.xml')
        self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1*u.nanometer)
        integrator = mm.LangevinIntegrator(310, 0.05, 100 * mm.unit.femtosecond)

        # Add forces
        print('Adding forces...')
        self.add_forcefield()
        print('Forces added ;)\n')

        # Minimize energy
        print('Minimizing energy...')
        platform = mm.Platform.getPlatformByName(self.platform)
        simulation = Simulation(pdb.topology, self.system, integrator, platform)
        simulation.reporters.append(StateDataReporter(stdout, (self.N_steps*sim_step)//10, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
        simulation.reporters.append(DCDReporter(self.path+'/other/stochastic_LE.dcd', 5))
        simulation.context.setPositions(pdb.positions)
        current_platform = simulation.context.getPlatform()
        print(f"Simulation will run on platform: {current_platform.getName()}")
        simulation.minimizeEnergy(tolerance=0.001)

        # self.state = simulation.context.getState(getPositions=True)
        # PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.path+'/pdbs/MDLE_0.cif', 'w'))
        print('Energy minimization done :D\n')

        # Run molecular dynamics simulation
        if run_MD:
            print('Running molecular dynamics (wait for 10 steps)...')
            start = time.time()
            heats = list()
            for i in range(1,self.N_steps):
                for nn in range(self.N_coh):
                    # Update Loop Extrusion Force
                    le_force = self.LE_forces[nn]
                    le_force.setBondParameters(i-1, self.M[nn,i-1], self.N[nn,i-1], 0.1, 0.0)
                    le_force.setBondParameters(i, self.M[nn,i], self.N[nn,i], 0.1, 3e5)
                    le_force.updateParametersInContext(simulation.context)

                    # Update repli potential which is linked with replication fork propagation
                    if i>=self.t_rep and i<self.t_rep+15*self.rep_duration:
                        locs1 = np.nonzero(self.l_forks[:,(i-self.t_rep)%15])[0]
                        locs2 = np.nonzero(self.r_forks[:,(i-self.t_rep)%15])[0]
                        locs = np.union1d(locs1,locs2)
                        for j in range(0,self.N_beads):
                            self.external_force.setParticleParameters(j,j+self.N_beads,[1000.0])
                        for l in locs:
                            self.repli_force.setBondParameters(int(l),int(l),int(l)+self.N_beads,[0.0])
                        self.repli_force.updateParametersInContext(simulation.context)
                        self.external_force.updateParametersInContext(simulation.context)
                    elif i>=self.t_rep+15*self.rep_duration:
                        for j in range(self.N_beads):
                            self.external_force.setParticleParameters(j, j+self.N_beads, [0.0])
                        self.external_force.updateParametersInContext(simulation.context)
                
                simulation.step(sim_step)
                if i%self.step==0 and i>self.burnin*self.step:
                    self.state = simulation.context.getState(getPositions=True)
                    if write_files: PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.path+f'/pdbs/MDLE_{i//self.step-self.burnin}.cif', 'w'))
                    save_path = self.path+f'/heatmaps/heat_{i//self.step-self.burnin}.svg' if write_files else None
                    heats.append(get_heatmap(self.state.getPositions(),save_path=save_path,save=write_files))
                    time.sleep(5)
            end = time.time()
            elapsed = end - start

            print(f'Everything is done! Simulation finished succesfully!\nMD finished in {elapsed/60:.2f} minutes.\n')

            self.avg_heat = np.average(heats,axis=0)
            self.std_heat = np.std(heats,axis=0)
            np.save(self.path+f'/other/avg_heatmap.npy',self.avg_heat)
            np.save(self.path+f'/other/std_heatmap.npy',self.std_heat)
            if plots:
                figure(figsize=(10, 10))
                plt.imshow(self.avg_heat,cmap="Reds",vmax=1)
                plt.colorbar()
                plt.savefig(self.path+f'/plots/avg_heatmap.svg',format='svg',dpi=500)
                plt.savefig(self.path+f'/plots/avg_heatmap.pdf',format='pdf',dpi=500)
                # plt.colorbar()
                plt.close()

                figure(figsize=(10, 10))
                plt.imshow(self.std_heat,cmap="Reds",vmax=1)
                plt.colorbar()
                plt.savefig(self.path+f'/plots/std_heatmap.svg',format='svg',dpi=500)
                plt.savefig(self.path+f'/plots/std_heatmap.pdf',format='pdf',dpi=500)
                # plt.colorbar()
                plt.close()
                
                # heats_to_prob(self.heats,self.path+f'/plots/prob_dist_heatmap.svg',burnin=self.burnin,q=0.1)
            return self.avg_heat
    
    def add_forcefield(self):
        '''
        Here is the definition of the forcefield.

        There are the following energies:
        - ev force: repelling LJ-like forcefield
        - harmonic bond force: to connect adjacent beads.
        - angle force: for polymer stiffness.
        - LE forces: this is a list of force objects. Each object corresponds to a different cohesin. It is needed to define a force for each time step.
        '''
        # Leonard-Jones potential for excluded volume
        ev_force = mm.CustomNonbondedForce('epsilon*((sigma1+sigma2)/(r+r_small))^3')
        ev_force.addGlobalParameter('epsilon', defaultValue=10)
        ev_force.addGlobalParameter('r_small', defaultValue=0.01)
        ev_force.addPerParticleParameter('sigma')
        for i in range(2*self.N_beads):
            ev_force.addParticle([0.05])
        self.system.addForce(ev_force)

        # Harmonic bond borce between succesive beads
        self.bond_force = mm.HarmonicBondForce()
        for i in range(self.N_beads - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 300000.0)
        for i in range(self.N_beads,2*self.N_beads - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 300000.0)
        self.system.addForce(self.bond_force)

        # Harmonic angle force between successive beads so as to make chromatin rigid
        self.angle_force = mm.HarmonicAngleForce()
        for i in range(self.N_beads - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 400)
        for i in range(self.N_beads,2*self.N_beads - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 400)
        self.system.addForce(self.angle_force)
        
        # LE force that connects cohesin restraints
        self.LE_forces = list()
        for nn in tqdm(range(self.N_coh)):
            LE_force = mm.HarmonicBondForce()
            for i in range(self.N_steps):
                if i==0:
                    LE_force.addBond(self.M[nn,i], self.N[nn,i], 0.1, 50000.0)
                    LE_force.addBond(self.N_beads+self.M[nn,i], self.N_beads+self.N[nn,i], 0.1, 3e5)
                else:
                    LE_force.addBond(self.M[nn,i], self.N[nn,i], 0.1, 0)
                    LE_force.addBond(self.N_beads+self.M[nn,i], self.N_beads+self.N[nn,i], 0.1, 0)
            self.system.addForce(LE_force)
            self.LE_forces.append(LE_force)

        # Replication force to bring together the two polymers
        self.repli_force = mm.CustomBondForce('D * (r-r0)')
        self.repli_force.addGlobalParameter('r0', defaultValue=0.1)
        self.repli_force.addPerBondParameter('D')
        for i in range(self.N_beads):
            self.repli_force.addBond(i, i + self.N_beads, [5e3])
        self.system.addForce(self.repli_force)

        # Custom External Force
        # Define a random direction for the force
        random_direction = np.random.randn(3)
        random_direction /= np.linalg.norm(random_direction)  # Normalize the direction vector
        self.external_force = mm.CustomExternalForce(f'F * (ax * x + ay * y + az * z)')
        self.external_force.addPerParticleParameter('F')
        self.external_force.addGlobalParameter('ax',defaultValue=random_direction[0])
        self.external_force.addGlobalParameter('ay',defaultValue=random_direction[1])
        self.external_force.addGlobalParameter('az',defaultValue=random_direction[2])
        for i in range(self.N_beads,2*self.N_beads):
            self.external_force.addParticle(i,[0.0])
        self.system.addForce(self.external_force)

def main():
    # A potential example
    M = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ms.npy')
    N = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ns.npy')
    md = MD_LE(4*M,4*N,2000,5,1)
    md.run_pipeline(write_files=False,plots=True,sim_step=100)
