import numpy as np
from beliefmatching import BeliefMatching
from pathlib import Path
import sys

sys.path.append("../")
from src.simulations import SurfaceCodeSim


def main():

    # error rate and number of shots
    p = 1e-3
    n = int(1e8)

    # where to save
    save_path = Path("../data/belief_matching_p1e-3_d_3_5_7_9_dt_3_4_5_6_7_8_9_10_11")

    code_sizes = [3, 5, 7, 9]
    rounds = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    data = np.zeros((4, 9))
    for i, d in enumerate(code_sizes):
        for j, r in enumerate(rounds):
            sim = SurfaceCodeSim(r, d, p, n)
            circuit = sim.circuit
            bm = BeliefMatching(circuit, max_bp_iters=20)

            shots, observables, n_trivial = sim.sample_syndromes()
            preds = bm.decode_batch(shots)
            n_correct = np.sum(preds == observables[:, None])
            failure_rate = (n - n_correct - n_trivial) / n
            data[i, j] = failure_rate

    np.save(save_path.parent / (save_path.name + ".npy"), data)
    np.savetxt(
        save_path.parent / (save_path.name + ".csv"),
        data,
        delimiter=",",
    )
    return 0


if __name__ == "__main__":
    main()
