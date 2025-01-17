import numpy as np
import stim
import pymatching
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from pathlib import Path
import sys

def decode(code_sz, reps, n=int(1e3), p=1e-3):

    circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=reps,
            distance=code_sz,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )
    sampler = circuit.compile_detector_sampler()
    matcher = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    
    stim_data, observable_flips = sampler.sample(shots=n, separate_observables=True)

    # sums over the detectors to check if we have a parity change
    shots_w_flips = np.sum(stim_data, axis=1) != 0
    n_trivial = np.invert(shots_w_flips).sum()

    # save only data for measurements with non-empty syndromes
    # but count how many trival (identity) syndromes we have
    shots = stim_data[shots_w_flips, :]
    flips = observable_flips[shots_w_flips, 0].astype(np.uint8)

    preds = matcher.decode_batch(shots)
    n_correct = np.sum(preds == flips[:, None]) 

    return {(code_sz, reps): n_correct}, {(code_sz, reps): n_trivial}

def main():
    
    # where to save
    save_path = Path("../data/pymatching_parallel")
    
    # error rate and number of shots
    p = 1e-3
    n = int(1e5)
    n_per_batch = int(1e3)
    n_batches = int(n / n_per_batch)
    
    # initialise sims and decoders
    code_sizes = [3]
    rounds = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # create dictionary labels used later
    labels = []
    for sz in code_sizes:
        for r in rounds:
            labels.append((sz, r))
    
    all_code_sz = np.array([[sz] * len(rounds) for sz in code_sizes]).flatten()
    all_rounds = np.array(rounds * 2)
    
    all_code_sz = np.tile(all_code_sz, n_batches)
    all_rounds = np.tile(all_rounds, n_batches)
    p = np.ones_like(all_rounds) * p
    n_per_batch = np.ones_like(all_rounds) * n_per_batch
    
    with Pool(processes=(cpu_count() - 1)) as pool:
        res = pool.starmap(decode, list(zip(all_code_sz, all_rounds, n_per_batch, p)))

    # collect the statistics
    correct = {}
    trivial = {}
    for c, t in res:
        c_key = list(c.keys())[0]
        c_val = list(c.values())[0]
        t_key = list(t.keys())[0]
        t_val = list(t.values())[0]
        
        if c_key in correct:
            correct[c_key] += c_val
        else: 
            correct.update(c)
        
        if t_key in trivial:
            trivial[t_key] += t_val
        else:
            trivial.update(t)
    
    data = np.zeros((len(code_sizes), len(rounds)))
    row_map = {sz: i for i, sz in enumerate(code_sizes)}
    col_map = {r: i for i, r in enumerate(rounds)}
    
    for l in labels:
        n_correct = correct[l]
        n_trivial = trivial[l]
        
        logical_accuracy = (n - n_correct - n_trivial) / n
        sz = l[0]
        rep = l[1]
        data[row_map[sz], col_map[rep]] = logical_accuracy
    
    np.save(save_path.parent / (save_path.name + ".npy"), data)
    np.savetxt(
        save_path.parent / (save_path.name + ".csv"),
        data,
        delimiter=",",
    )
    return 0
        
if __name__ == "__main__":
    main()
