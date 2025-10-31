import numpy as np
import matplotlib.pyplot as plt


def get_matrix_value(matrix, to_return, top=5):
    matrix_1d = matrix.flatten()
    args = matrix_1d.argsort()[1441:][1::2]
    if '.' in to_return:
        place, to_return = to_return.split('.')
        if to_return == 'best':
            idx_1d = args[int(place)-1]
        elif to_return == 'worst':
            idx_1d = args[-int(place)]
        x_idx, y_idx = np.unravel_index(idx_1d, matrix.shape)
    else:
        if to_return == 'best':
            idx_1d = args[:int(top)]
        elif to_return == 'worst':
            idx_1d = np.flip(args[-int(top):])
        x_idx, y_idx = np.unravel_index(idx_1d, matrix.shape)
    return x_idx, y_idx

def print_matrix_values(matrix, top=5):
    worst_x_idx, worst_y_idx = get_matrix_value(matrix, 'worst', top)
    print("\nWorst results:")
    for x, y, in zip(worst_x_idx, worst_y_idx):
        print(str(x) + ', ' + str(y) + " with " + str(matrix[x][y]))
    best_x_idx, best_y_idx = get_matrix_value(matrix, 'best', top)
    print("\nBest results:")
    for x, y, in zip(best_x_idx, best_y_idx):
        print(str(x) + ', ' + str(y) + " with " + str(matrix[x][y]))

def calc_diff_matrix(matrix1, matrix2):
    #matrix = abs(np.subtract(matrix1, matrix2))
    matrix = np.add(matrix1, matrix2)
    matrix = np.divide(matrix, np.max(matrix))
    return matrix

def calc_mean_matrix(matrix_list):
    result_matrix = matrix_list[0]
    for i in range(1, len(matrix_list)):
        result_matrix = np.add(result_matrix, matrix_list[i])
    result_matrix = np.divide(result_matrix, len(matrix_list))
    return result_matrix

def calc_row_mean_matrix(matrix):
    matrix = np.matrix(matrix)
    n = matrix.shape[0]
    summed_matrix = matrix.sum(axis=1)
    func = np.vectorize(lambda x: x/n)
    avg_list = func(summed_matrix)
    return np.array(avg_list)

def plot_heatmap(matrix, dataset, dimred=None, diff=False, img=False, mean_param_list=[]):
    fig, ax = plt.subplots(figsize=(10, 8))
    dataset_id = dataset.id
    if dimred:
        dimred_id = dimred.id
        if diff:
            ax.set_title("DIST difference heatmap evaluation of " + dimred_id + " on " + dataset_id)
        else:
            if mean_param_list:
                ax.set_title("DIST heatmap evaluation of " + dimred_id + " on " + dataset_id
                        + " with Parameters=" + ', '.join(map(str, map(int, mean_param_list))))
            else:
                ax.set_title("DIST heatmap evaluation of " + dimred_id + " on " + dataset_id
                        + " with Parameter=" + str(dimred.parameter))
    else:
       if img:
            ax.set_title("Heatmap evaluation of " + dataset_id + " images")
       else:
            ax.set_title("DIST heatmap evaluation of high dim. " + dataset_id)
    matrix[0][0] = 1
    ax.imshow(matrix, cmap='hot_r')
    ax.xaxis.tick_top()
    plt.show()

if __name__ == "__main__":
    pass