import pickle
from isolation import Isolation
import numpy as np


if __name__ == "__main__":
    #init_cells = Isolation().actions()
    # Strategy 1
    MODEL_STRAT_1 = 'good_books'
    init_cells = [43, 14, 87, 30, 114, 23, 56, 69, 47, 60]

    for cell in init_cells:
        cell_result = pickle.load(open("./" + MODEL_STRAT_1 + "/" + str(cell) + "test.pickle", "rb"))
        print(str(cell), ",", np.mean(list(map(lambda x: x[1], cell_result)))/100,
              np.mean(list(map(lambda x: x[1], cell_result[0:10]))) / 100,
              cell_result[0][1]/100, cell_result[0][0])

    #res = pickle.load(open("summary.pickle", "rb"))
    #for r in res:
    #    print(str(r[0]) + "," + str(r[1]))

