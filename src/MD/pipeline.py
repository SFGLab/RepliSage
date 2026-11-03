"""MD pipeline: initial structure -> minimization -> one structure per MC sample."""
import time
import numpy as np
import openmm as mm
from openmm.app import PDBxFile, ForceField, Simulation, DCDReporter
from .structures import compute_init_struct
from .md_plots import plot_md_energies
from ..io.structure_io import write_mmcif, write_cif, write_chimerax_script, generate_psf
from ..logger import log, fmt_time

class PipelineMixin:
    def build_initial_structure(self, init_struct):
        '''Initial structure(s) written as mmCIF + PSF. Second chain only with replication.'''
        log.info(f'Building initial structure ({init_struct}, {self.n_chains} chain(s))...')
        points1 = compute_init_struct(self.N_beads, init_struct)
        points2 = points1 + [0.2, 0.2, 0.2] if self.run_repli else None
        d = self.out_path + '/metadata/md_dynamics/'
        write_mmcif(points1, points2, d + 'LE_init_struct.cif')
        generate_psf(self.N_beads, d + 'replisage.psf', duplicated=self.run_repli)
        return PDBxFile(d + 'LE_init_struct.cif')

    def build_simulation(self, pdb, ff_path, integrator_mode, md_temperature, integrator_step):
        '''System, forces, integrator and OpenMM simulation object.'''
        self.system = ForceField(ff_path).createSystem(pdb.topology, nonbondedCutoff=2 * self.rw_l)
        if integrator_mode not in ('langevin', 'brownian'): raise ValueError("INTEGRATOR_TYPE must be 'langevin' or 'brownian'.")
        log.info("Building the force field and the OpenMM context (a moment)...")
        self.add_forcefield(self.burnin)
        self.simulation = self.make_simulation(pdb, integrator_mode, md_temperature, integrator_step)
        self.simulation.context.setPositions(pdb.positions)

    def platform_order(self):
        """Platforms to try: the requested GPU first, then the other GPU platform, CPU last (an explicit CPU stays CPU)."""
        want = (self.platform or "auto").strip().lower()
        if want == "cpu": return ["CPU"]
        first = {"cuda": "CUDA", "opencl": "OpenCL"}.get(want)
        return [first] + [x for x in ("CUDA", "OpenCL") if x != first] + ["CPU"] if first else ["CUDA", "OpenCL", "CPU"]

    def make_simulation(self, pdb, integrator_mode, md_temperature, integrator_step):
        """First platform that initialises: CUDA, then OpenCL, then CPU (any error moves on to the next one)."""
        props_gpu = {'DeviceIndex': str(self.device)} if self.device not in (None, '') else {}
        failed = []
        for name in self.platform_order():
            try:
                platform = mm.Platform.getPlatformByName(name)
                cls = mm.LangevinIntegrator if integrator_mode == 'langevin' else mm.BrownianIntegrator   # a fresh integrator per attempt
                ints = cls(md_temperature, 0.1 / mm.unit.femtosecond, integrator_step)
                sim = Simulation(pdb.topology, self.system, ints, platform, props_gpu if name != "CPU" else {})
                self.platform = name
                log.success(f"MD simulation will run on the {name} platform." + (f" ({'; '.join(failed)})" if failed else ""))
                return sim
            except Exception as e:
                failed.append(f"{name} unavailable: {str(e).splitlines()[0][:80]}")
        raise RuntimeError("No OpenMM platform could be initialised: " + "; ".join(failed))

    def minimize_chunked(self, tol, label, chunk=100, max_chunks=200):
        """Energy minimization in chunks of `chunk` iterations with a live bar (potential energy, RMS force); stops once the RMS force is below tol."""
        ctx, n = self.simulation.context, self.n_chains * self.N_beads
        bar = log.progress(total=max_chunks, desc=label, unit="chunk")
        for _ in range(max_chunks):
            self.simulation.minimizeEnergy(tolerance=tol, maxIterations=chunk)
            st = ctx.getState(getEnergy=True, getForces=True)
            e = st.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)
            f = np.sqrt(np.mean(np.sum(st.getForces(asNumpy=True).value_in_unit(mm.unit.kilojoule_per_mole / mm.unit.nanometer) ** 2, axis=1)))
            bar.update(1); bar.set_postfix_str(f"E = {e:,.0f} kJ/mol, RMS force = {f:,.1f}")
            if f < tol: break
        bar.close()

    def minimize_initial(self, pdb, tol, reporters, sim_step):
        '''Initial energy minimization (live energy bar); saves the minimized model.'''
        log.note(f"WAITING for the initial energy minimization to converge (RMS force < {tol}, at most {200 * 100} iterations). This can take several minutes on a CPU; the bar below shows the energy.")
        t = time.time(); self.minimize_chunked(tol, "Minimization")
        log.success(f"Initial energy minimization finished in {fmt_time(time.time() - t)}")
        self.state = self.simulation.context.getState(getPositions=True)
        if reporters:
            self.simulation.reporters.append(DCDReporter(self.out_path + '/metadata/md_dynamics/replisage.dcd', sim_step))
        with open(self.out_path + '/metadata/md_dynamics/minimized_model.cif', 'w') as fh:
            PDBxFile.writeFile(pdb.topology, self.state.getPositions(), fh)

    def separation_threshold(self):
        '''Distance (nm) between chain centres above which sister chromatids count as separated (ellipsoid major axis).'''
        Rg = np.sqrt(self.N_beads * 0.1 ** 2 / 6)
        return 2 * np.sqrt(5 / 3) * Rg

    def temperature_of(self, kinetic):
        '''T = 2*KE / (N_df * kB), no constraints.'''
        kB = (mm.unit.BOLTZMANN_CONSTANT_kB * mm.unit.AVOGADRO_CONSTANT_NA).value_in_unit(mm.unit.kilojoule_per_mole / mm.unit.kelvin)
        return 2 * kinetic / (3 * self.n_chains * self.N_beads * kB)

    def run_pipeline(self, init_struct='rw', tol=1.0, sim_step=100, reporters=False, mode='MD', integrator_mode='langevin',
                     p_ev=0.01, md_temperature=310 * mm.unit.kelvin, integrator_step=10.0 * mm.unit.femtosecond,
                     ff_path='forcefields/classic_sm_ff.xml', plot_energy=True):
        '''
        Run the molecular simulation pipeline; writes one .cif per recorded MC sample into ensemble/.
        mode: 'MD' (minimize once, then dynamics) or 'EM' (minimize at every sample).
        Returns the sample at which the sister chromatids separated (None without replication / no separation).
        '''
        if mode not in ('MD', 'EM'): raise ValueError('Mode can be only EM or MD.')
        self.p_ev = p_ev
        pdb = self.build_initial_structure(init_struct)
        self.build_simulation(pdb, ff_path, integrator_mode, md_temperature, integrator_step)
        self.minimize_initial(pdb, tol, reporters, sim_step)

        thr = self.separation_threshold()
        if self.run_repli: log.diag(f"Separation threshold: {thr:.2f} nm (ellipsoid major axis).")
        separation_started, fully_separated_step, distance = False, None, np.nan
        energies, kinetics, temps, times, labels = [], [], [], [], []
        t0 = time.time()
        n_s = self.N_steps - self.burnin
        log.note(f"Running the 3D simulation: {n_s} structures, " + (f"{sim_step} MD steps each." if mode == 'MD' else "one energy minimization each.") + " This is the slow part; the bar shows the energy.")
        pbar = log.progress(total=self.N_steps - self.burnin, desc=f'MD ({mode}) structures')

        for i in range(self.burnin, self.N_steps):
            rep_per = self.compute_rep(i)
            if self.run_repli: self.change_repliforce(i)
            self.change_loop(i)
            if p_ev > 0: self.ps_ev = np.random.rand(self.N_beads)
            self.change_ev()
            if self.has_comps: self.change_comps(i)

            if mode == 'MD': self.simulation.step(sim_step)
            else: self.simulation.minimizeEnergy(tolerance=tol)
            label, idx = self.add_label(i); labels.append(label)

            if self.run_repli and i * self.step == self.t_rep + self.rep_duration:   # start pulling the sisters apart
                self.change_pull_force(); separation_started = True

            self.state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            fn = f'ensemble_{i - self.burnin + 1}_{label}.cif'
            write_cif(self.state.getPositions(asNumpy=True).value_in_unit(mm.unit.angstrom), self.n_chains, self.out_path + '/ensemble/' + fn)
            if i == self.burnin: write_chimerax_script(self.out_path + '/ensemble', [fn], self.n_chains)

            kin = self.state.getKineticEnergy().value_in_unit(mm.unit.kilojoule_per_mole)
            energies.append(self.state.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole))
            kinetics.append(kin); temps.append(self.temperature_of(kin))
            times.append((i - self.burnin + 1) * sim_step * integrator_step.value_in_unit(mm.unit.picosecond))

            if self.run_repli and separation_started and fully_separated_step is None:
                pos = self.state.getPositions(asNumpy=True).value_in_unit(mm.unit.nanometer)
                distance = np.linalg.norm(pos[:self.N_beads].mean(axis=0) - pos[self.N_beads:].mean(axis=0))
                if distance > thr:
                    fully_separated_step = i; self.disable_pull_force()
            pbar.update(1); pbar.set_postfix_str(f'replicated {rep_per:.0f}%, E = {energies[-1]:,.0f} kJ/mol, T = {temps[-1]:.0f} K')
        pbar.close()

        if self.run_repli:
            if fully_separated_step is not None: log.success(f"Polymers fully separated at step {fully_separated_step} (distance: {distance:.2f} nm)")
            else: log.alert("Polymers did not fully separate during the simulation. Consider running more simulation steps.")
        log.success(f"MD finished in {fmt_time(time.time() - t0)}")
        log.table("MD summary", [("mode", mode), ("chains", self.n_chains), ("beads per chain", self.N_beads),
                                 ("structures written", self.N_steps - self.burnin),
                                 ("mean potential energy (kJ/mol)", f"{np.mean(energies):.1f}"),
                                 ("mean temperature (K)", f"{np.mean(temps):.1f}")])
        if plot_energy: plot_md_energies(self.out_path, times, energies, kinetics, temps, labels)
        return fully_separated_step
