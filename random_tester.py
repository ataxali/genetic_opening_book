import pickle
from isolation import Isolation
import numpy as np
from opening_book import GenomeTester, NEG_INF_INT
from multiprocessing.dummy import Pool as ThreadPool
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor
import mpi4py
from mpi4py import MPI
import socket
import sys


N_CORES = 2
N_THREADS = N_CORES * 2
dir_name = "random"
N_OPPONENTS = 100
mpi4py.rc.recv_mprobe = False


def win_rate_per_genome(init_cell, genome, search_depth=3):
    # splice genes on threads
    scores = []
    for _ in range(N_OPPONENTS):
        scores.append(GenomeTester(init_cell, genome, search_depth).run())
    win_count = sum(map(lambda x: x[1] > NEG_INF_INT, scores))
    win_rate = 100. * win_count / N_OPPONENTS
    return win_rate


def test_cell_genomes(init_cell):
    print("Child proc running on:", socket.gethostname(), "working on", str(init_cell), "...")
    sys.stdout.flush()
    result = win_rate_per_genome(init_cell, None, 3)
    print("Child proc running on:", socket.gethostname(), "finished on",
          str(init_cell), ":", result, "...")
    sys.stdout.flush()
    return (init_cell, result)


def main():
    init_cells = Isolation().actions()

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is None: return # worker process
        results = list(executor.map(test_cell_genomes, init_cells, chunksize=1))

    print("Main proc done...")
    sys.stdout.flush()

    pickle.dump(results, open("./" + dir_name + "/" + "summary.pickle", "wb"))


if __name__ == "__main__":
    main()



