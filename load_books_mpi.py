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
N_OPPONENTS = 100
dir_name = "temp"
mpi4py.rc.recv_mprobe = False


def win_rate_per_genome(init_cell, genome, search_depth=3):
    # splice genes on threads
    scores = []
    for _ in range(N_OPPONENTS):
        scores.append(GenomeTester(init_cell, genome, search_depth).run())
    win_count = sum(map(lambda x: x[1] > NEG_INF_INT, scores))
    win_rate = 100. * win_count / N_OPPONENTS
    return genome, win_rate


def test_cell_genomes(tup_cell_genomes):
    init_cell, genomes = tup_cell_genomes
    print("Child proc running on:", socket.gethostname(), "working on", str(init_cell), "...")
    sys.stdout.flush()
    results = []
    for genome in genomes:
        results.append(win_rate_per_genome(init_cell, genome))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    print("Child proc running on:", socket.gethostname(), "finished on",
          str(init_cell), "...")
    sys.stdout.flush()
    pickle.dump(list(results),
                open("./" + dir_name + "/" + str(init_cell) + "test.pickle", "wb"))
    return init_cell, results


def main():
    #init_cells = Isolation().actions()
    init_cells = [94, 43, 14, 62, 23, 56, 92, 106, 70, 57]
    tasks = []

    for cell in init_cells:
        res = pickle.load(open("./" + dir_name + "/" + str(cell) + "res.pickle", "rb"))
        gen_results = list(map(lambda x: x[1], res))
        final_genomes = list(map(lambda x: x[0], gen_results[-1]))
        tasks.append((cell, final_genomes))
        #idx = np.random.choice(len(final_genomes), 500)
        #reduced_genomes = [final_genomes[i] for i in idx]
        #tasks.append((cell, reduced_genomes))

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is None: return # worker process
        results = list(executor.map(test_cell_genomes, tasks, chunksize=1))

    print("Main proc done...")
    sys.stdout.flush()

    pickle.dump(results, open("./" + dir_name + "/" + "summary.pickle", "wb"))


if __name__ == "__main__":
    main()



