import stim
import numpy as np


class QECCodeSim:
    def __init__(self, repetitions, distance, p, n_shots, code_task, seed):
        self.distance = distance
        self.repetitions = repetitions
        self.p = p
        self.n_shots = n_shots

        self.circuit = stim.Circuit.generated(
            code_task,
            rounds=repetitions,
            distance=distance,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )

        self.compiled_sampler = self.circuit.compile_detector_sampler(seed=seed)

    def get_detector_coords(self):
        # create detection grid for circuit
        det_coords = self.circuit.get_detector_coordinates()
        det_coords = np.array(list(det_coords.values()))

        # rescale space like coordinates:
        det_coords[:, :2] = det_coords[:, :2] / 2

        # convert to integers
        det_coords = det_coords.astype(np.uint8)

        return det_coords

    def sample_syndromes(self):
        stim_data, observable_flips = self.compiled_sampler.sample(
            shots=self.n_shots, separate_observables=True,
        )
        # sums over the detectors to check if we have a parity change
        shots_w_flips = np.sum(stim_data, axis=1) != 0

        # save only data for measurements with non-empty syndromes
        stabilizer_changes = stim_data[shots_w_flips, :]
        flips = observable_flips[shots_w_flips, 0]
        
        return stabilizer_changes, flips.astype(np.uint8)


class RepetitionCodeSim(QECCodeSim):
    def __init__(self, repetitions, distance, p, n_shots, seed=None):
        super().__init__(repetitions, distance, p, n_shots, "repetition_code:memory")


# TODO: Make it possible to create syndrome masks in one go instead if looping through shots
class SurfaceCodeSim(QECCodeSim):
    def __init__(
        self,
        repetitions,
        distance,
        p,
        n_shots,
        code_task="surface_code:rotated_memory_z",
        seed=None,
    ):
        super().__init__(repetitions, distance, p, n_shots, code_task, seed)

    def syndrome_mask(self):
        sz = self.distance + 1

        syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
        syndrome_x[::2, 1 : sz - 1 : 2] = 1
        syndrome_x[1::2, 2::2] = 1

        syndrome_z = np.rot90(syndrome_x) * 3

        return np.dstack([syndrome_x + syndrome_z] * (self.repetitions + 1))

    def generate_syndromes(self, n_syndromes=None):
        
        det_coords = super().get_detector_coords()
        stabilizer_changes, flips = super().sample_syndromes()
        mask = self.syndrome_mask()

        syndromes = []
        for cycle in stabilizer_changes:
            syndrome = np.zeros_like(mask)

            # stack stabilizer changes as first-to-last time step
            # note that detector coords represents the coordinates (x, y, t) of the measurement qubits
            syndrome[det_coords[:, 1], det_coords[:, 0], det_coords[:, 2]] = cycle

            # we only care about differences in measurements
            syndrome[:, :, 1:] = (syndrome[:, :, 1:] - syndrome[:, :, 0:-1]) % 2

            # using our code_grid we can convert X/Z stabilizer measurements to 1:s and 3:s
            syndrome[np.nonzero(syndrome)] = mask[np.nonzero(syndrome)]
            syndromes.append(syndrome)
            
        # make sure we get enough syndromes if a certain number is desired
        if n_syndromes is not None:
            while len(syndromes) < n_syndromes:
                new_syndromes, new_flips = self.generate_syndromes()
                syndromes += new_syndromes
                flips =  np.concatenate((flips, new_flips))

            syndromes = syndromes[:n_syndromes]
            flips = flips[:n_syndromes]
        return syndromes, flips
