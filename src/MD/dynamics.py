"""Time-dependent updates of the forces during the MD."""
import numpy as np
import openmm as mm

class DynamicsMixin:
    """Methods that change forces from one MC sample to the next."""

    def change_ev(self):
        '''
        Update excluded volume force parameters.
        '''
        ev_strength = (self.ps_ev > self.p_ev).astype(int) * np.sqrt(200) if self.p_ev > 0 else np.sqrt(200) * np.ones(self.N_beads)
        for n in range(self.N_beads):
            self.ev_force.setParticleParameters(n, [ev_strength[n], 0.05])
        if self.run_repli:  # Only update replicated particles if replication is enabled
            for n in range(self.N_beads, 2 * self.N_beads):
                self.ev_force.setParticleParameters(n, [ev_strength[n % self.N_beads], 0.05])
        self.ev_force.updateParametersInContext(self.simulation.context)

    def change_repliforce(self, i):
        if not self.run_repli:
            return  # Skip if replication is disabled

        if i * self.step >= self.t_rep and i * self.step < self.t_rep + self.rep_duration:
            rep_dna = self.replicated_dna[i * self.step - self.t_rep]   # time-major matrix
            rep_locs = np.nonzero(rep_dna)[0]
            for l in rep_locs:
                self.repli_force.setBondParameters(int(l), int(l), int(l) + self.N_beads, [self.rw_l / 4, 1000])
        elif i * self.step >= self.t_rep + self.rep_duration:
            for j in range(self.N_beads):
                self.repli_force.setBondParameters(j, j, j + self.N_beads, [5 * self.rw_l, 0.0])
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

    def change_pull_force(self):
        for n in range(self.N_beads):
            self.pull_force.setParticleParameters(n,n, [1000.0 * mm.unit.kilojoule_per_mole / mm.unit.nanometer])
        if self.run_repli:
            for n in range(self.N_beads,2*self.N_beads):
                self.pull_force.setParticleParameters(n,n, [-1000.0 * mm.unit.kilojoule_per_mole / mm.unit.nanometer])
        self.pull_force.updateParametersInContext(self.simulation.context)

    def disable_pull_force(self):
        for n in range(self.N_beads):
            self.pull_force.setParticleParameters(n,n, [0.0 * mm.unit.kilojoule_per_mole / mm.unit.nanometer])
        if self.run_repli:
            for n in range(self.N_beads,2*self.N_beads):
                self.pull_force.setParticleParameters(n,n, [0.0 * mm.unit.kilojoule_per_mole / mm.unit.nanometer])
        self.pull_force.updateParametersInContext(self.simulation.context)
