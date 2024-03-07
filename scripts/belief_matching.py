import numpy as np
from beliefmatching import BeliefMatching
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from pathlib import Path
import sys

sys.path.append("../")
from src.simulations import SurfaceCodeSim

def decode(code_sz, reps, n=int(1e3), p=1e-3):

    sim = SurfaceCodeSim(reps, code_sz, p, n)
    bm = BeliefMatching(sim.circuit, max_bp_iters=20)

    shots, observables, n_trivial = sim.sample_syndromes()
    preds = bm.decode_batch(shots)
    n_correct = np.sum(preds == observables[:, None]) 

    return {(code_sz, reps): n_correct}, {(code_sz, reps): n_trivial}

def main():
    
    # where to save
    save_path = Path("../data/bm_parallel")
    
    # error rate and number of shots
    p = 1e-3
    n = int(1e4)
    n_per_batch = int(1e3)
    n_batches = int(n / n_per_batch)
    
    # initialise sims and decoders
    code_sizes = [7]
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
