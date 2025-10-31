from DataSet import DataSet
from DimRed import DimRed
from EvalCrit import EvalCrit
import utils
import sys
import numpy as np
import matplotlib.pyplot as plt

def open_first_coil20_imgs(matrix, to_search):
    coil20 = DataSet('COIL20')
    for i in range(1,coil20.datapoints*coil20.datapoints):
        if to_search == 'same':
            to_calc = str(i) + '.worst'
            label1, _, label2, _ = coil20.calc_img_info(matrix, to_calc)
            if label1 == label2:
                break
        elif to_search == 'diff':
            to_calc = str(i) + '.best'
            label1, _, label2, _ = coil20.calc_img_info(matrix, to_calc)
            if label1 != label2:
                break
    open_coil20_imgs(matrix, to_calc)

def open_coil20_imgs(matrix, to_open):
    coil20 = DataSet('COIL20')
    label1, img_num1, label2, img_num2 = coil20.calc_img_info(matrix, to_open)
    coil20.open_coil20_img(label1, img_num1)
    coil20.open_coil20_img(label2, img_num2)

def diff_heatmap_creator(dataset, dimred, plot=False):
    best_matrix = heatmap_creator(dataset, dimred, 'best')
    worst_matrix = heatmap_creator(dataset, dimred, 'worst')
    matrix = utils.calc_diff_matrix(best_matrix, worst_matrix)
    utils.print_matrix_values(matrix)
    if plot:
        utils.plot_heatmap(matrix, dataset, dimred, diff=True)
    return matrix

def heatmap_creator(dataset, dimred, to_return='', plot=False):
    dimred.apply(dataset)
    evalcrit = EvalCrit('DIST', dataset, dimred)
    evalcrit.eval(verbose=False)
    param_list = []
    if '-' in to_return:
        from_num, to_return = to_return.split('-')
        to_num, to_return = to_return.split('.')
        matrix_list = []
        for i in range(int(from_num), int(to_num)+1):
            param, _ = evalcrit.get_top_eval(str(i) + '.' + to_return)
            param_list.append(param)
            index = dimred.get_range().index(param)
            data_low = dimred.data_low[index]
            matrix_list.append(evalcrit.eval_dist(data_low, return_matrix='beforediff'))
        matrix = utils.calc_mean_matrix(matrix_list)
    else:
        if '.' in to_return:
            param, _ = evalcrit.get_top_eval(to_return)
        else:
            if not(dimred.id == 'MDS'):
                param = int(to_return)
        dimred = dimred.transform_method(new_param=param)
        evalcrit.set_dimred(dimred)
        matrix = evalcrit.eval_dist(dimred.data_low, return_matrix='beforediff')
    utils.print_matrix_values(matrix)
    if plot:
        utils.plot_heatmap(matrix, dataset, dimred, mean_param_list=param_list)
    return matrix

def high_dim_heatmap_creator(dataset, plot=False):
    matrix = dataset.get_pairwise_distances_high_dim()
    utils.print_matrix_values(matrix)
    if plot:
        utils.plot_heatmap(matrix, dataset)
    return matrix

def reverse_engineering(dataset, dimred, param=False, normalize=True):
    dimred.apply(dataset)
    evalcrit = EvalCrit('DIST', dataset, dimred)
    evalcrit.eval(verbose=False)
    if param:
        dimred = dimred.transform_method(new_param=param)
        evalcrit.set_dimred(dimred)
    matrix = evalcrit.eval_dist(dimred.data_low, return_matrix='diff')
    avg_row = utils.calc_row_mean_matrix(matrix)
    if normalize:
        saved_data_high = dataset.data_high
        dataset.data_high = np.append(dataset.data_high, [[0, 0, 0]], axis=0)
        avg_row = np.append(avg_row, [[0.5]], axis=0)
    dataset.plot_data_high(avg_row, plot=False)
    dataset.plot_data_low(dimred, plot=False)
    plt.show()
    if normalize:
        dataset.data_high = saved_data_high

def merge_data_low_results(dataset, dimreds):
    results = []
    param_ranges = []
    data_lows = []
    for dimred in dimreds:
        param_ranges.extend(list(dimred.get_range()))
        data_lows.extend(dataset.load_data_low(dimred))
    results = sorted(zip(param_ranges, data_lows))
    new_param_range = [x[0] for x in results]
    new_data_low = [x[1] for x in results]
    new_dimred = DimRed(dimreds[0].id, parameter_range=(new_param_range[0], new_param_range[-1]+1, 1))
    dataset.save_data_low(new_dimred, new_data_low)

def merge_eval_results(evalcrits):
    results = []
    param_ranges = []
    eval_results = []
    for evalcrit in evalcrits:
        param_ranges.extend(list(evalcrit.dimred.get_range()))
        eval_results.extend(evalcrit.load_eval())
    results = sorted(zip(param_ranges, eval_results))
    new_param_range = [x[0] for x in results]
    new_eval_results = [x[1] for x in results]
    new_dimred = DimRed(evalcrits[0].dimred.id, parameter_range=(new_param_range[0], new_param_range[-1]+1, 1))
    new_evalcrit = EvalCrit(evalcrits[0].criterion, evalcrits[0].dataset, new_dimred)
    new_evalcrit.save_eval(new_eval_results)

def plot_low(dataset, dimred, plot_parameter=None):
    dimred.apply(dataset)
    if plot_parameter:
        dimred = dimred.transform_method(new_param=plot_parameter)
    dataset.plot_data_low(dimred)
    print(dimred.data_low)

def plot_eval(criterion, dataset, dimred, plot_param_range=None):
    dimred.apply(dataset)
    if plot_param_range:
        dimred = dimred.transform_method(new_param_range=plot_param_range)
    evalcrit = EvalCrit(criterion, dataset, dimred)
    evalcrit.eval(verbose=True)
    if dimred.parameter_range:
        evalcrit.sort_print_eval(10, order='ASC')
        evalcrit.sort_print_eval(10, order='DESC')
        evalcrit.plot_eval()
    else:
        print(evalcrit.eval_result)
        evalcrit.plot_eval()

def plot_multiple_evals(criterion, dataset, dimreds, plot_param_range=None):
    evalcrits = []
    for dimred in dimreds:
        dimred.apply(dataset)
        if plot_param_range:
            dimred = dimred.transform_method(new_param_range=plot_param_range)
        evalcrit = EvalCrit(criterion, dataset, dimred)
        evalcrit.eval(verbose=False)
        evalcrits.append(evalcrit)
    EvalCrit.plot_multiple_evals(evalcrits)

def plot_multiple_evals_beforeNafter(criterion, dataset_before, dataset_after, dimreds_before, dimred_after, plot_param_range=None):
    evalcrits = []
    for dimred in dimreds_before:
        dimred.apply(dataset_before)
        if plot_param_range:
            dimred = dimred.transform_method(new_param_range=plot_param_range)
        if dimred.id == 'MDS': dimred.plot_color = '#73E7C5'
        if dimred.id == 'LLE': dimred.plot_color = '#FEBA86'
        if dimred.id == 'TSNE': dimred.plot_color = '#B1AFD5'
        evalcrit = EvalCrit(criterion, dataset_before, dimred)
        evalcrit.eval(verbose=False)
        evalcrits.append(evalcrit)
    for dimred in dimred_after:
        dimred.apply(dataset_after)
        if plot_param_range:
            dimred = dimred.transform_method(new_param_range=plot_param_range)
        evalcrit = EvalCrit(criterion, dataset_after, dimred)
        evalcrit.eval(verbose=False)
        evalcrits.append(evalcrit)
    EvalCrit.plot_multiple_evals(evalcrits)

def plot_nn_evolution(criterions, dataset, dimred, plot_param_range=None):
    evalcrits = []
    init_dimred = dimred
    for criterion in criterions:
        dimred.apply(dataset)
        if plot_param_range:
            dimred = dimred.transform_method(new_param_range=plot_param_range)
        evalcrit = EvalCrit(criterion, dataset, dimred)
        evalcrit.eval(verbose=False)
        evalcrits.append(evalcrit)
        dimred = init_dimred
    EvalCrit.plot_nn_evolution(evalcrits)


''' Slurm Batch Processing for data_low '''
# input = int(sys.argv[1])
# dataset = DataSet('trans_Mammoth')
# dimred = DimRed('TSNE', parameter_range=(input, 10000, 999))
# dimred.apply(dataset)

''' data_low Result Merging '''
# dataset = DataSet('trans_Mammoth')
# dimred_list = []
# for n in range(2,1001):
#     dimred_list.append(DimRed('TSNE', parameter_range=(n,10000,999)))
# merge_data_low_results(dataset, dimred_list)

''' Slurm Batch Processing for eval '''
# input = int(sys.argv[1])
# dataset = DataSet('trans_Mammoth')
# dimred = DimRed('LLE', parameter_range=(2,3334,1))
# dimred.apply(dataset)
# dimred = dimred.transform_method(save_old=False, new_param_range=(input, 3334, 333))
# evalcrit = EvalCrit('DIST', dataset, dimred)

''' eval Result Merging '''
# for i in range(13):
#     evalcrit_list = []
#     for n in range(2,1001):
#         dataset = DataSet('trans_Mammoth')
#         dimred = DimRed('TSNE', parameter_range=(n,10000,999))
#         evalcrit = EvalCrit(str(2**i)+'NN', dataset, dimred)
#         evalcrit_list.append(evalcrit)
#     merge_eval_results(evalcrit_list)

''' DataSets '''
coil20 = DataSet('COIL20')
mammoth = DataSet('Mammoth')
trans_mammoth = DataSet('trans_Mammoth')

''' DimReds '''
mds_B = DimRed('MDS')
lle_coil = DimRed('LLE', parameter_range=(2,1440,1))
lle_dense = DimRed('LLE', parameters=[584,1196,1227], variant='dense')
tsne_coil = DimRed('TSNE', parameter_range=(2,1440,1))
random_tsne = DimRed('TSNE', parameter_range=(1340,1440,1), variant='random')
lle_mam_B = DimRed('LLE', parameter_range=(2,10000,1))
tsne_mam_B = DimRed('TSNE', parameter_range=(2,10000,1))

mds_A = DimRed('MDS')
tsne_mam_A = DimRed('TSNE', parameter_range=(2,10000,1))
trans_lle_A = DimRed('LLE', parameter_range=(2,3334,1))

''' Matrices '''
#matrix_diff = diff_heatmap_creator(coil20, tsne_coil)
#matrix_high_coil20 = high_dim_heatmap_creator(coil20, plot=False)

''' Plotter '''
#mammoth.plot_data_high()
#trans_mammoth.plot_data_high()
#plot_low(trans_mammoth, mds_A)

#plot_low(trans_mammoth, trans_lle_A, plot_parameter=1788)
#plot_low(trans_mammoth, tsne_mam_A, plot_parameter=197)
#plot_low(mammoth, tsne_mam_B, plot_parameter=197)
#plot_low(coil20, mds_B)
#plot_low(coil20, lle_coil, plot_parameter=3)
#plot_low(coil20, tsne_coil, plot_parameter=1390)
plot_eval('1NN', mammoth, lle_mam_B)
#plot_eval('512NN', coil20, tsne_coil)
#plot_eval('DIST', coil20, tsne_coil)
#plot_eval('1NN', coil20, lle_coil, plot_param_range=(500,600,1))
#lot_eval('DIST', coil20, random_tsne, plot_param_range=(1340,1440,1))
#plot_eval('ATA', coil20, mds)
# plot_eval('ATA', coil20, lle_coil)
# plot_eval('ATA', coil20, tsne_coil)
#plot_multiple_evals('DIST', trans_mammoth, [mds_A, trans_lle_A, tsne_mam_A])
#plot_multiple_evals('DIST', mammoth, [mds_B, lle_mam_B, tsne_mam_B])
#plot_multiple_evals('1NN', coil20, [mds_B, lle_coil, tsne_coil])
#plot_multiple_evals_beforeNafter('DIST', mammoth, trans_mammoth, [mds_B, lle_mam_B, tsne_mam_B], [mds_A, trans_lle_A, tsne_mam_A], plot_param_range=(2,800,1))
#plot_multiple_evals_beforeNafter('1NN', mammoth, trans_mammoth, [lle_mam_B], [trans_lle_A])
#plot_nn_evolution([str(2**x) + 'NN' for x in np.flip(range(13))], mammoth, tsne_mam)

#high_dim_heatmap_creator(mammoth, plot=True)
#heatmap_creator(mammoth, mds, plot=True)

#heatmap_creator(mammoth, mds_B, plot=True)
#heatmap_creator(coil20, tsne_coil, '1390', plot=True)
#heatmap_creator(coil20, tsne_coil, '1418', plot=True)
#heatmap_creator(coil20, lle_coil, '1229', plot=True)
#heatmap_creator(coil20, lle_coil, '1.best', plot=True)
#heatmap_creator(coil20, tsne_coil, '1.best', plot=True)
# heatmap_creator(coil20, tsne,'worst', plot=True)

#diff_heatmap_creator(coil20, mds_B, plot=True)
# diff_heatmap_creator(coil20, lle, plot=True)
#diff_heatmap_creator(coil20, tsne_coil, plot=True)

#open_coil20_imgs(matrix_high_coil20, '1.best')

#open_first_coil20_imgs(matrix_high_coil20, 'diff')

#coil20.plot_image_diff('diff')

# reverse_engineering(mammoth, mds_B, normalize=False)
# reverse_engineering(mammoth, lle_mam_B, 2463, normalize=False)
# reverse_engineering(mammoth, lle_mam_B, 5, normalize=False)
# reverse_engineering(mammoth, tsne_mam_B, 9736, normalize=False)
# reverse_engineering(mammoth, tsne_mam_B, 9932, normalize=False)