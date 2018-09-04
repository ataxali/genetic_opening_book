import sys
import random
import pickle
from isolation import Isolation, DebugState
from multiprocessing.dummy import Pool as ThreadPool
from numpy.random import choice
import socket
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor
from mpi4py import MPI
import mpi4py


N_CORES = 2
N_THREADS = N_CORES * 2
NEG_INF_INT = -100000
mpi4py.rc.recv_mprobe = False



def build_rand_genomes(n_genomes, init_cell, genome_len=4):
    assert (init_cell <= 114) and (init_cell >= 0), "Invalid opening cell value"
    genomes = []
    for i in range(n_genomes):
        genome = []
        board = Isolation().result(init_cell)
        for j in range(genome_len):
            try:
                board = Isolation(board=board.board, ply_count=board.ply_count + 1,
                                  locs=board.locs)
                next_action = random.choice(board.actions())
                genome.append(next_action)
                board = board.result(next_action)
            except Exception as e:
                # try random sample from previous genomes, otherwise skip genome
                if len(genomes) > 0:
                    genome = random.choice(genomes)
                else:
                    break
        if len(genome) == genome_len:
            genomes.append(genome)
    return genomes


class GenomeTester:

    def __init__(self, init_cell, genome, search_depth):
        assert (init_cell <= 114) and (init_cell >= 0), "Invalid opening cell value"
        self.init_cell = init_cell
        self.board = Isolation()
        self.player0_moves = 0
        self.player1_moves = 0
        self.genome = genome
        self.search_depth = search_depth
        self.active_player = 0
        self.move_history = []

    def run(self):
        ############################
        ####### mini max  ##########
        def minimax(state, depth, player_id):

            def min_value(state, depth, player_id):
                if state.terminal_test(): return state.utility(player_id)
                if depth <= 0: return score(state, player_id)
                value = float("inf")
                for action in state.actions():
                    value = min(value,
                                max_value(state.result(action), depth - 1,
                                          player_id))
                return value

            def max_value(state, depth, player_id):
                if state.terminal_test(): return state.utility(player_id)
                if depth <= 0: return score(state, player_id)
                value = float("-inf")
                for action in state.actions():
                    value = max(value,
                                min_value(state.result(action), depth - 1,
                                          player_id))
                return value

            return max(state.actions(),
                       key=lambda x: min_value(state.result(x), depth - 1,
                                               player_id))

        def score(state, player_id):
            own_loc = state.locs[player_id]
            opp_loc = state.locs[1 - player_id]
            own_liberties = state.liberties(own_loc)
            opp_liberties = state.liberties(opp_loc)
            return len(own_liberties) - len(opp_liberties)
        ####### mini max  ##########
        ############################

        if self.player0_moves == 0:
            self.board = self.board.result(self.init_cell)

        if self.player1_moves == 0:
            self.board = self.board.result(random.choice(self.board.actions()))

        while not self.board.terminal_test():
            if self.active_player == 0:
                if self.player0_moves < len(self.genome):
                    next_move = self.genome[self.player0_moves]
                    if next_move not in self.board.actions():
                        # move is most likely blocked (not as bad as a loss)
                        #return self.genome, NEG_INF_INT
                        next_move = minimax(self.board, self.search_depth,
                                            player_id=0)
                else:
                    next_move = minimax(self.board, self.search_depth,
                                        player_id=0)
                self.player0_moves += 1
                self.active_player = 1
            else:
                next_move = minimax(self.board, self.search_depth,
                                    player_id=1)
                self.player1_moves += 1
                self.active_player = 0
            self.board = self.board.result(next_move)
            if self.player0_moves < len(self.genome):
                self.move_history.append(next_move)

        player0_score = self.board.utility(player_id=0)
        if player0_score < 0: # lost
            return self.genome, float("-inf")
        elif player0_score == 0: # game didnt finish
            return self.genome, NEG_INF_INT
        else:
            return self.genome, -1.0 * (self.player0_moves + self.player1_moves)


class GenerationTester:
    def __init__(self, n_genomes, init_cell, genome_len, search_depth, learning_rate,
                 mutate_rate):
        self.n_genomes = n_genomes
        self.init_cell = init_cell
        self.genome_len = genome_len
        self.search_depth = search_depth
        try:
            self.genomes = build_rand_genomes(n_genomes, init_cell, genome_len)
        except Exception as e:
            self.genomes = None
            print("Failed to build genomes for cell " + str(init_cell) + " -> " + str(e))
        self.learning_rate = learning_rate
        self.mutate_rate = mutate_rate
        self.gen_count = 0

    def run(self, gen_limit=10):
        results = []

        if self.genomes is None:
            return results
        elif len(self.genomes) == 0:
            return results

        for i in range(gen_limit):
            if i == gen_limit-1:
                results.append(self.run_one_generation(do_mutate=False))
            else:
                results.append(self.run_one_generation(do_mutate=True))
        return results

    def run_one_generation(self, do_mutate=True):
        #print("Running generation", self.gen_count, "for cell", str(self.init_cell) +
        #      "..."); sys.stdout.flush()

        ## Generate results for parent generation
        pool = ThreadPool(N_THREADS)
        results = pool.imap(lambda x: GenomeTester(self.init_cell, x,
                                                   self.search_depth).run(),
                            self.genomes)
        pool.close()
        pool.join()
        results = sorted(results, key=lambda x: x[1])
        win_count = sum(map(lambda x: x[1] > NEG_INF_INT, results))
        #print("Generation " + str(self.gen_count) + " won {:.1f}% of matches".format(
        #    100. * win_count / len(self.genomes)) + " (opt score " + str(results[-1][1]) + ")...")
        #sys.stdout.flush()


        ## Throw away lowest genomes
        n_deprecated = int(self.learning_rate * len(self.genomes))
        results = results[n_deprecated:]
        #print("Throwing away", str(n_deprecated), "worst genomes..."); sys.stdout.flush()

        ## Create child generation
        def splice_genomes(genomes, init_cell, weighted_probs, genome_len,
                           max_attempts=1000):
            is_done = False
            new_genome = []
            attempt = 0
            while not is_done:
                attempt += 1
                parent_1, parent_2 = choice(a=range(len(genomes)), size=2,
                                            replace=False, p=weighted_probs)
                genome0 = genomes[parent_1]
                genome1 = genomes[parent_2]
                new_genome = []
                for i in range(0, genome_len, 2):
                    new_genome.append(genome0[i])
                    new_genome.append(genome1[i+1])
                board = Isolation().result(init_cell)
                valid_genome = True
                for action in new_genome:
                    board = Isolation(board=board.board,
                                      ply_count=board.ply_count + 1,
                                      locs=board.locs)
                    if action not in board.actions():
                        valid_genome = False
                        break
                    board = board.result(action)
                is_done = valid_genome and (len(new_genome) == genome_len)
                is_done = True if attempt >= max_attempts else is_done
            return new_genome

        # calculate sampling probabilties
        weighted_probs = list(map(lambda x: 1.0/(-1*x[1]) if x[1] > NEG_INF_INT
                                  else 1.0/(-1*NEG_INF_INT), results))
        prob_factor = 1 / sum(weighted_probs)
        weighted_probs = [prob_factor * p for p in weighted_probs]
        self.genomes = list(map(lambda x: x[0], results)).copy()

        # splice genes on threads
        pool = ThreadPool(N_THREADS)
        spliced_genomes = pool.imap(lambda x: splice_genomes(self.genomes, self.init_cell,
                                                             weighted_probs, self.genome_len),
                                    range(n_deprecated))
        pool.close()
        pool.join()
        spliced_genomes = list(spliced_genomes)

        # add spliced genes to next gene pool
        n_spliced_genomes = 0
        while (len(self.genomes) < self.n_genomes) and spliced_genomes:
            new_genome = spliced_genomes.pop()
            if new_genome:
                self.genomes.append(new_genome)
                n_spliced_genomes += 1
        #print("Adding", str(n_spliced_genomes), "to genome pool..."); sys.stdout.flush()

        # mutate gene pool for next generation
        if do_mutate:
            n_mutations = int(self.mutate_rate * len(self.genomes))
            #print("Mutating", n_mutations, "genomes in pool...")
            mutate_idxs = choice(a=range(len(self.genomes)), size=n_mutations, replace=False)
            for mutate_idx in mutate_idxs:
                try:
                    genome = self.genomes[mutate_idx].copy()
                    genome_idx = random.randint(0, self.genome_len-1)
                    board = Isolation().result(self.init_cell)
                    for i in range(genome_idx):
                        action = genome[i]
                        board = Isolation(board=board.board,
                                          ply_count=board.ply_count + 1,
                                          locs=board.locs)
                        board = board.result(action)
                    for i in range(genome_idx, len(genome)):
                        board = Isolation(board=board.board, ply_count=board.ply_count + 1, locs=board.locs)
                        mutation_action = random.choice(board.actions())
                        board = board.result(mutation_action)
                        genome[i] = mutation_action
                    self.genomes[mutate_idx] = genome[0:self.genome_len]
                except Exception as e:
                    pass  # do nothing if mutation fails

        self.gen_count += 1
        return 100. * win_count / len(self.genomes), results


def book_process(init_cell):
    print("Proc on", socket.gethostname(), "working on", str(init_cell) + "...")
    #################
    n_genomes = 2000
    gen_limit = 50
    #################

    genome_len = 10
    search_depth = 3
    learning_rate = 0.25
    mutate_rate = 0.05

    res = GenerationTester(n_genomes, init_cell, genome_len, search_depth,
                           learning_rate, mutate_rate).run(gen_limit)
    #print("[" + str(init_cell) + "] Win rates by generation:",
    #      list(map(lambda x: x[0], res)))
    #print("[" + str(init_cell) + "] Optimal score by generation:",
    #      list(map(lambda x: x[1][-1][1], res)))
    pickle.dump(res,
                open("./results_len10_gen50/" + str(init_cell) + "res.pickle", "wb"))
    return res


def main():
    tasks = Isolation().actions()
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is None: return  # worker process
        _ = list(executor.map(book_process, tasks, chunksize=1))
    return 1


if __name__ == "__main__":
    main()
    print("Main proc done...")
    sys.stdout.flush()

