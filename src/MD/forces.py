"""OpenMM force terms of the chromatin model."""
import numpy as np
import openmm as mm

class ForcesMixin:
    """Definition of the forcefield (backbone, loops, replication, compartments)."""

    def add_evforce(self):
        'Leonard-Jones potential for excluded volume'
        self.ev_force = mm.CustomNonbondedForce(f'(epsilon1*epsilon2*(sigma1*sigma2)/(r+r_small))^3')
        self.ev_force.addGlobalParameter('r_small', defaultValue=0.01)
        self.ev_force.addPerParticleParameter('sigma')
        self.ev_force.addPerParticleParameter('epsilon')
        self.ev_force.setCutoffDistance(distance=0.2)
        self.ev_force.setForceGroup(1)
        for i in range(self.N_beads):
            self.ev_force.addParticle([np.sqrt(200), 0.05])
        if self.run_repli:
            for i in range(self.N_beads, 2 * self.N_beads):
                self.ev_force.addParticle([np.sqrt(200), 0.05])
        self.system.addForce(self.ev_force)

    def add_bonds(self):
        'Harmonic bond borce between succesive beads'
        self.bond_force = mm.HarmonicBondForce()
        self.bond_force.setForceGroup(0)
        for i in range(self.N_beads - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 5e5)
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads - 1):
                self.bond_force.addBond(i, i + 1, 0.1, 5e5)
        self.system.addForce(self.bond_force)

    def add_stiffness(self):
        'Harmonic angle force between successive beads so as to make chromatin rigid'
        self.angle_force = mm.HarmonicAngleForce()
        self.angle_force.setForceGroup(0)
        for i in range(self.N_beads - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 200)
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads - 2):
                self.angle_force.addAngle(i, i + 1, i + 2, np.pi, 200)
        self.system.addForce(self.angle_force)

    def add_loops(self, i):
        'LE force that connects cohesin restraints or adds dummy loops if undefined'
        self.LE_force = mm.HarmonicBondForce()
        self.LE_force.setForceGroup(0)

        for n in range(self.N_coh):
            m_val = self.M[n, i]
            n_val = self.N[n, i]

            if m_val >= 0 and n_val >= 0:
                # Add real loop
                self.LE_force.addBond(m_val, n_val, 0.1, 5e4)
                if self.run_repli:
                    self.LE_force.addBond(self.N_beads + m_val, self.N_beads + n_val, 0.1, 5e4)
            else:
                # Add dummy loop with zero strength between random monomers
                r1 = np.random.randint(0, self.N_beads)
                r2 = np.random.randint(0, self.N_beads)
                self.LE_force.addBond(r1, r2, 0.1, 0.0)
                if self.run_repli:
                    self.LE_force.addBond(self.N_beads + r1, self.N_beads + r2, 0.1, 0.0)

        self.system.addForce(self.LE_force)

    def add_repliforce(self,i):
        'Replication force to bring together the two polymers'
        self.repli_force = mm.CustomBondForce('D * (r-r0)^2')
        self.repli_force.setForceGroup(0)
        self.repli_force.addPerBondParameter('r0')
        self.repli_force.addPerBondParameter('D')
        
        if i*self.step<self.t_rep:
            for i in range(self.N_beads):
                self.repli_force.addBond(i, i + self.N_beads, [0.0,5e4])
        elif i*self.step>=self.t_rep and i*self.step<self.t_rep+self.rep_duration:
            rep_dna = self.replicated_dna[i*self.step-self.t_rep]   # time-major matrix
            rep_locs = np.nonzero(rep_dna)[0]
            for i in range(self.N_beads):
                if i in rep_locs:
                    self.repli_force.addBond(i, i + self.N_beads, [self.rw_l/4,1000])
                else:
                    self.repli_force.addBond(i, i + self.N_beads, [0.0,5e4])
        else:
            for i in range(self.N_beads):
                self.repli_force.addBond(i, i + self.N_beads, [5*self.rw_l,0.0])
        
        self.system.addForce(self.repli_force)

    def add_pulling_force(self):
        self.pull_force = mm.CustomExternalForce("-f * x")
        self.pull_force.addPerParticleParameter("f")
        for i in range(self.N_beads):
            self.pull_force.addParticle(i, [0.0 * mm.unit.kilojoule_per_mole / mm.unit.nanometer])
        if self.run_repli:
            for i in range(self.N_beads, 2 * self.N_beads):
                self.pull_force.addParticle(i, [0.0 * mm.unit.kilojoule_per_mole / mm.unit.nanometer])
        self.system.addForce(self.pull_force)

    def add_blocks(self,i):
        'Block copolymer forcefield for the modelling of compartments.'
        cs = self.Cs[:,i] if self.Cs.ndim>1 else self.Cs
        self.comp_force = mm.CustomNonbondedForce('delta(c1-c2)*E*exp(-(r-r0)^2/(2*sigma^2)); E=Ea1*delta(s1-2)*delta(s2-2)+Ea2*delta(s1-1)*delta(s2-1)+Eb1*delta(s1)*delta(s2)+Eb2*delta(s1+1)*delta(s2+1)+Eb3*delta(s1+2)*delta(s2+2)')
        self.comp_force.setForceGroup(1)
        self.comp_force.addGlobalParameter('sigma',defaultValue=self.rw_l/2)
        self.comp_force.addGlobalParameter('r0',defaultValue=0.2)
        self.comp_force.addGlobalParameter('Ea1',defaultValue=-0.2)
        self.comp_force.addGlobalParameter('Ea2',defaultValue=-0.4)
        self.comp_force.addGlobalParameter('Eb1',defaultValue=-0.6)
        self.comp_force.addGlobalParameter('Eb2',defaultValue=-0.8)
        self.comp_force.addGlobalParameter('Eb3',defaultValue=-1.0)
        # self.comp_force.setCutoffDistance(distance=self.rw_l)
        self.comp_force.addPerParticleParameter('s')
        self.comp_force.addPerParticleParameter('c')
        for i in range(self.N_beads):
            self.comp_force.addParticle([cs[i],self.chain_idx[i]])
        if self.run_repli:
            for i in range(self.N_beads,2*self.N_beads):
                self.comp_force.addParticle([cs[i%self.N_beads],self.chain_idx[i]])
        self.system.addForce(self.comp_force)

    def add_container(self, R=10.0, C=100.0):
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

    def add_forcefield(self, i, use_container=False):
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
        self.add_pulling_force()
        if use_container: self.add_container(R=self.rw_l)
        if self.has_comps: self.add_blocks(i)
        if self.run_repli:  # Add replication forces only if replication is enabled
            self.add_repliforce(i)
        self.add_loops(i)
