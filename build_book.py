import pickle
from isolation import Isolation
import numpy as np


if __name__ == "__main__":
    init_cells = Isolation().actions()
    #data = dict()
    # Strategy 1
    #model_data = []
    #for cell in init_cells:
    #    cell_result = pickle.load(open("./" + 'results_len10_gen50' + "/" + str(cell) + "test.pickle", "rb"))
    #    #mean_win_rate = np.mean(list(map(lambda x: x[1], cell_result[0:10])))
    #    model_data.append((cell, cell_result[0][0], cell_result[0][1]))
    #sorted_model_data = sorted(model_data, key=lambda x: x[2], reverse=True)
    #data['results_len10_gen50'] = list(map(lambda x: (x[0], x[1]), sorted_model_data)).copy()

    # Strategy 2
    model_data = []
    init_cells = [43, 14, 87, 30, 114, 23, 56, 69, 47, 60]
    data_dict = dict()

    for cell in init_cells:
        cell_result = pickle.load(open("./" + 'good_books' + "/" + str(cell) + "test.pickle", "rb"))
        #mean_win_rate = np.mean(list(map(lambda x: x[1], cell_result[0:10])))
        model_data.append((cell, cell_result[0][0], cell_result[0][1]))
        data_dict[cell] = (cell, cell_result[0][0])
    #sorted_model_data = sorted(model_data, key=lambda x: x[2], reverse=True)
    #data['results_len4_gen50'] = list(map(lambda x: (x[0], x[1]), sorted_model_data))

    with open("data.pickle", 'wb') as f:
        pickle.dump(data_dict, f)