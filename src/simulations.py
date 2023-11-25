import stim
import numpy as np

from src.graph_representation import get_3D_graph

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

    def sample_syndromes(self, n_shots=None):
        if n_shots == None:
            n_shots = self.n_shots
        stim_data, observable_flips = self.compiled_sampler.sample(
            shots=n_shots,
            separate_observables=True,
        )
        
        # sums over the detectors to check if we have a parity change
        shots_w_flips = np.sum(stim_data, axis=1) != 0
        n_trivial_syndromes = np.invert(shots_w_flips).sum()
        
        # save only data for measurements with non-empty syndromes
        # but count how many trival (identity) syndromes we have
        stabilizer_changes = stim_data[shots_w_flips, :]
        flips = observable_flips[shots_w_flips, 0]
        
        ######################################
        # flips = observable_flips[shots_w_flips]
        # WAS UINT8!
        #######################################

        return stabilizer_changes, flips.astype(np.uint8), n_trivial_syndromes


class RepetitionCodeSim(QECCodeSim):
    def __init__(
        self,
        repetitions,
        distance,
        p,
        n_shots,
        seed=None,
    ):
        super().__init__(
            repetitions, distance, p, n_shots, "repetition_code:memory", seed
        )


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

    def generate_syndromes(self, n_syndromes=None, n_shots=None):
        det_coords = super().get_detector_coords()
        stabilizer_changes, flips, n_trivial_preds = super().sample_syndromes(n_shots)

        mask = np.repeat(
            self.syndrome_mask()[None, ...], stabilizer_changes.shape[0], 0
        )
        syndromes = np.zeros_like(mask)
        syndromes[
            :, det_coords[:, 1], det_coords[:, 0], det_coords[:, 2]
        ] = stabilizer_changes
        
        syndromes[..., 1:] = (syndromes[..., 1:] - syndromes[..., 0:-1]) % 2
        syndromes[np.nonzero(syndromes)] = mask[np.nonzero(syndromes)]

        # make sure we get enough non-trivial syndromes if a certain number is desired
        if n_syndromes is not None:
            while syndromes.shape[0] < n_syndromes:
                n_shots = n_syndromes - len(syndromes)
                new_syndromes, new_flips, new_n_trivial_preds = self.generate_syndromes(n_shots=n_shots)
                syndromes = np.concatenate((syndromes, new_syndromes))
                flips = np.concatenate((flips, new_flips))
                n_trivial_preds += new_n_trivial_preds

            syndromes = syndromes[:n_syndromes]
            flips = flips[:n_syndromes]

        return syndromes, flips, n_trivial_preds

    def generate_batch(
        self,
        m_nearest_nodes,
        power,
    ):
        """
        Generates a batch of graphs from a list of stim experiments.
        """
        batch = []
        stim_data_list = []
        observable_flips_list = []
        
        stim_data, observable_flips, n_trivial = self.sample_syndromes(self.n_shots)
        mask = self.syndrome_mask()
        detector_coordinates = self.get_detector_coords()
        
        stim_data_list.extend(stim_data[:self.n_shots])
        observable_flips_list.extend(observable_flips[:self.n_shots])
        
        for i in range(len(stim_data_list)):
            # convert to syndrome grid:
            syndrome = self.stim_to_syndrome_3D(mask, detector_coordinates, stim_data_list[i])
            # get the logical equivalence class:
            true_eq_class = np.array([int(observable_flips_list[i])])
            # map to graph representation
            graph = get_3D_graph(
                syndrome_3D=syndrome,
                target=true_eq_class,
                power=power,
                m_nearest_nodes=m_nearest_nodes,
                use_knn=False,
            )
            batch.append(graph)
        return batch, n_trivial
    
    def stim_to_syndrome_3D(self, mask, coordinates, stim_data):
        """
        Converts a stim detection event array to a syndrome grid.
        1 indicates a violated X-stabilizer, 3 a violated Z stabilizer.
        Only the difference between two subsequent cycles is stored.
        """
        # initialize grid:
        syndrome_3D = np.zeros_like(mask)

        # first to last time-step:
        syndrome_3D[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]] = stim_data

        # only store the difference in two subsequent syndromes:
        syndrome_3D[:, :, 1:] = (syndrome_3D[:, :, 1:] - syndrome_3D[:, :, 0:-1]) % 2

        # convert X (Z) stabilizers to 1(3) entries in the matrix
        syndrome_3D[np.nonzero(syndrome_3D)] = mask[np.nonzero(syndrome_3D)]

        return syndrome_3D