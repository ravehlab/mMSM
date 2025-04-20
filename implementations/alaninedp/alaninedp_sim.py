import numpy as np
import openmm as omm
import openmm.app as ommapp
import openmm.unit as ommunits
from multiprocessing import Pool
from mmsm.mmsm_base.base_trajectory_sampler import BaseTrajectorySampler


class DialanineOMMSampler(BaseTrajectorySampler):
    def __init__(self, prmtop_path, dt_ps=0.002, temp0=400, cuda=False, concurrent_sims=1, return_vs=False):
        self.prmtop_path = prmtop_path
        self.prmtop = ommapp.AmberPrmtopFile(prmtop_path)
        self.system = self.prmtop.createSystem(nonbondedCutoff=2 * omm.unit.nanometer,
                                          constraints=ommapp.HBonds, implicitSolvent=ommapp.GBn2)
        self.dt = dt_ps
        self.temp0 = temp0
        self.cuda = cuda
        self.concurrent = concurrent_sims
        self.start_state = None
        self.return_vs = return_vs
        if concurrent_sims == 1:
            self.integrator = self._get_integrator()
            if cuda:
                platform = omm.Platform.getPlatformByName('CUDA')
                self.simulation = ommapp.Simulation(self.prmtop.topology, self.system, self.integrator, platform)
                # self.simulation = ommapp.Simulation(self.prmtop.topology, self.system, self.integrator,
                #                                     omm.Platform.getPlatformByName('CUDA'), {'DeviceIndex': '0'})
            else:
                platform = omm.Platform.getPlatformByName('CPU')
                self.simulation = ommapp.Simulation(self.prmtop.topology, self.system, self.integrator, platform)
                # self.simulation = ommapp.Simulation(self.prmtop.topology, self.system, self.integrator)

    @property
    def timestep_size(self):
        return self.dt

    @property
    def initial_state(self):
        return self.start_state

    @initial_state.setter
    def initial_state(self, value):
        self.start_state = value

    def get_initial_sample(self, sample_len, n_samples, sample_interval=1):
        return self.sample_from_states([self.initial_state], sample_len, n_samples, sample_interval)

    def sample_from_states(self, states, sample_len, n_samples, sample_interval=1):
        """Will simulate sample_len-1 steps."""
        if self.concurrent > 1:
            return self._sample_from_states_parallel(states, sample_len, n_samples, sample_interval)

        trajs = []
        for start_state in states:

            if self.return_vs:
                start_pos_quan = self.array_to_quantity(start_state[0])
                start_vs_quan = self.array_to_quantity(start_state[1],
                                                       ommunits.nanometer / ommunits.picosecond)
            else:
                start_pos_quan = self.array_to_quantity(start_state)
                start_vs_quan = self.array_to_quantity(np.zeros_like(start_state),
                                          ommunits.nanometer / ommunits.picosecond)

            for samp in range(n_samples):
                traj = [start_state]
                self.simulation.context.setPositions(start_pos_quan)
                self.simulation.context.setVelocities(start_vs_quan)
                for i in range(sample_len - 1):
                    self.simulation.step(sample_interval)

                    if self.return_vs:
                        cur_state = self.simulation.context.getState(getPositions=True, getVelocities=True)
                        traj.append(np.array([cur_state.getPositions(asNumpy=True).value_in_unit(ommunits.angstrom),
                                    cur_state.getVelocities(asNumpy=True).value_in_unit(
                                        ommunits.nanometer / ommunits.picosecond)]))
                    else:
                        cur_state = self.simulation.context.getState(getPositions=True)
                        traj.append(cur_state.getPositions(asNumpy=True).value_in_unit(ommunits.angstrom))
                trajs.append(traj)
        return trajs

    def _sample_from_states_parallel(self, states, sample_len, n_samples, sample_interval=1):
        tasks = [(s, sample_len-1, sample_interval) for s in states for _ in range(n_samples)]
        with Pool(self.concurrent) as p:
            trajs = p.starmap(self._single_simulation, tasks)
        # for t in tasks:
        #     a = self._single_simulation(*t)
        return trajs

    def _single_simulation(self, start_state, n_steps, sample_interval=1):
        simulation = self._setup_simulation(start_state)
        traj = [start_state]
        for i in range(n_steps):
            simulation.step(sample_interval)
            if self.return_vs:
                cur_state = simulation.context.getState(getPositions=True, getVelocities=True)
                traj.append([cur_state.getPositions(asNumpy=True).value_in_unit(ommunits.angstrom),
                             cur_state.getVelocities(asNumpy=True).value_in_unit(
                                 ommunits.nanometer / ommunits.picosecond)])
            else:
                cur_state = simulation.context.getState(getPositions=True)
                traj.append(cur_state.getPositions(asNumpy=True).value_in_unit(ommunits.angstrom))
        return traj

    def _setup_simulation(self, start_state):
        prmtop = ommapp.AmberPrmtopFile(self.prmtop_path)
        system = prmtop.createSystem(nonbondedCutoff=2 * omm.unit.nanometer,
                                     constraints=ommapp.HBonds, implicitSolvent=ommapp.GBn2)

        integrator = self._get_integrator()

        if self.cuda:
            # If using CUDA, all simulations share the same GPU
            platform = omm.Platform.getPlatformByName('CUDA')
            # Optionally, you can manage device indices if multiple GPUs are available
            simulation = ommapp.Simulation(prmtop.topology, system, integrator, platform)
        else:
            # For CPU, limit each simulation to a single thread
            platform = omm.Platform.getPlatformByName('CPU')
            simulation = ommapp.Simulation(
                prmtop.topology, system, integrator,
                platform, {'Threads': '1'}
            )

        # simulation = ommapp.Simulation(prmtop.topology, system, integrator)
        if self.return_vs:
            start_pos_quan = self.array_to_quantity(start_state[0])
            start_vs_quan = self.array_to_quantity(start_state[1],
                                                   ommunits.nanometer / ommunits.picosecond)
        else:
            start_pos_quan = self.array_to_quantity(start_state)
            start_vs_quan = self.array_to_quantity(np.zeros_like(start_state),
                                                   ommunits.nanometer / ommunits.picosecond)
        simulation.context.setPositions(start_pos_quan)
        simulation.context.setVelocities(start_vs_quan)
        return simulation

    def _get_integrator(self):
        # return omm.BrownianIntegrator(self.temp0 * omm.unit.kelvin, 250 / omm.unit.picosecond,
        #                               self.dt * omm.unit.picoseconds)
        return omm.LangevinMiddleIntegrator(self.temp0 * omm.unit.kelvin, 0.01 / omm.unit.picosecond,
                                            self.dt * omm.unit.picoseconds)

    @staticmethod
    def array_to_quantity(array, unit=ommunits.angstrom):
        """Return a Quantity object with array as its value, in angstrom. Not sure how
        efficient this is - perhaps the simulations can accept a Quantity object defined differently."""
        return ommunits.Quantity([omm.Vec3(*row) for row in array.tolist()], unit=unit)

    @staticmethod
    def quantity_to_array(quantity):
        """Get the underlying array in the Quantity object."""
        return np.array(quantity.value_in_unit(quantity.unit))
