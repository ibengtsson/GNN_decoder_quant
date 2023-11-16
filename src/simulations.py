import stim
import numpy as np
import torch


# class QECCodeSim:
#     def __init__(self, repetitions, distance, p, n_shots, device, code_task, seed):
#         self.distance = distance
#         self.repetitions = repetitions
#         self.p = p
#         self.n_shots = n_shots
#         self.device = device

#         self.circuit = stim.Circuit.generated(
#             code_task,
#             rounds=repetitions,
#             distance=distance,
#             after_clifford_depolarization=p,
#             after_reset_flip_probability=p,
#             before_measure_flip_probability=p,
#             before_round_data_depolarization=p,
#         )

#         self.compiled_sampler = self.circuit.compile_detector_sampler(seed=seed)

#     def get_detector_coords(self):
#         # create detection grid for circuit
#         det_coords = self.circuit.get_detector_coordinates()
#         det_coords = torch.tensor(list(det_coords.values()))

#         # rescale space like coordinates:
#         det_coords[:, :2] = det_coords[:, :2] / 2

#         # convert to integers
#         det_coords = det_coords.long()

#         return det_coords

#     def sample_syndromes(self, n_shots=None):
#         if n_shots == None:
#             n_shots = self.n_shots
#         stim_data, observable_flips = self.compiled_sampler.sample(
#             shots=n_shots,
#             separate_observables=True,
#         )

#         # convert to torch tensors to utilize GPU
#         stim_data = torch.from_numpy(stim_data).to(self.device).int()
#         observable_flips = torch.from_numpy(observable_flips.astype(np.uint8)).to(self.device)

#         # sums over the detectors to check if we have a parity change
#         shots_w_flips = torch.sum(stim_data, axis=1) != 0

#         # save only data for measurements with non-empty syndromes
#         stabilizer_changes = stim_data[shots_w_flips, :]
#         flips = observable_flips[shots_w_flips, 0]

#         return stabilizer_changes, flips


# class RepetitionCodeSim(QECCodeSim):
#     def __init__(self, repetitions, distance, p, n_shots, device, seed=None):
#         super().__init__(
#             repetitions, distance, p, n_shots, device, "repetition_code:memory", seed
#         )

# class SurfaceCodeSim(QECCodeSim):
#     def __init__(
#         self,
#         repetitions,
#         distance,
#         p,
#         n_shots,
#         device,
#         code_task="surface_code:rotated_memory_z",
#         seed=None,
#     ):
#         super().__init__(repetitions, distance, p, n_shots, device, code_task, seed)

#     def syndrome_mask(self):
#         sz = self.distance + 1

#         syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
#         syndrome_x[::2, 1 : sz - 1 : 2] = 1
#         syndrome_x[1::2, 2::2] = 1

#         syndrome_z = np.rot90(syndrome_x) * 3

#         return np.dstack([syndrome_x + syndrome_z] * (self.repetitions + 1))

#     def generate_syndromes(self, n_syndromes=None, n_shots=None):
#         det_coords = super().get_detector_coords()
#         stabilizer_changes, flips = super().sample_syndromes(n_shots)

#         mask = np.repeat(
#             self.syndrome_mask()[None, ...], stabilizer_changes.shape[0], 0
#         )
#         mask = torch.from_numpy(mask).int().to(self.device)
#         syndromes = torch.zeros_like(mask)
#         syndromes[
#             :, det_coords[:, 1], det_coords[:, 0], det_coords[:, 2]
#         ] = stabilizer_changes
                
#         syndromes[..., 1:] = (syndromes[..., 1:] - syndromes[..., 0:-1]) % 2
#         syndromes[torch.nonzero(syndromes, as_tuple=True)] = mask[torch.nonzero(syndromes, as_tuple=True)]

#         # make sure we get enough syndromes if a certain number is desired
#         if n_syndromes is not None:
#             while syndromes.shape[0] < n_syndromes:
#                 n_shots = n_syndromes - len(syndromes)
#                 new_syndromes, new_flips = self.generate_syndromes(n_shots=n_shots)
#                 syndromes = torch.cat((syndromes, new_syndromes))
#                 flips = torch.cat((flips, new_flips))

#             syndromes = syndromes[:n_syndromes]
#             flips = flips[:n_syndromes]

#         return syndromes, flips

##########################################################
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

        # save only data for measurements with non-empty syndromes
        stabilizer_changes = stim_data[shots_w_flips, :]
        flips = observable_flips[shots_w_flips, 0]

        return stabilizer_changes, flips.astype(np.uint8)


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
        stabilizer_changes, flips = super().sample_syndromes(n_shots)

        mask = np.repeat(
            self.syndrome_mask()[None, ...], stabilizer_changes.shape[0], 0
        )
        syndromes = np.zeros_like(mask)
        syndromes[
            :, det_coords[:, 1], det_coords[:, 0], det_coords[:, 2]
        ] = stabilizer_changes
        
        syndromes[..., 1:] = (syndromes[..., 1:] - syndromes[..., 0:-1]) % 2
        syndromes[np.nonzero(syndromes)] = mask[np.nonzero(syndromes)]

        # make sure we get enough syndromes if a certain number is desired
        if n_syndromes is not None:
            while syndromes.shape[0] < n_syndromes:
                n_shots = n_syndromes - len(syndromes)
                new_syndromes, new_flips = self.generate_syndromes(n_shots=n_shots)
                syndromes = np.concatenate((syndromes, new_syndromes))
                flips = np.concatenate((flips, new_flips))

            syndromes = syndromes[:n_syndromes]
            flips = flips[:n_syndromes]

        return syndromes, flips
