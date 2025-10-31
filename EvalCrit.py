from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statistics


class EvalCrit():

    def __init__(self, criterion, dataset, dimred):
        self.criterion = criterion
        self.dataset = dataset
        self.dimred = dimred
        self.eval_result = np.array([])
        if criterion[-2:] == 'NN':
            self.k = int(criterion[:-2])
            self.order = 'DESC'
        elif criterion == 'DIST':
            self.order = 'ASC'
        elif criterion == 'ATA':
            pass
        else:
            raise Exception("specified criterion is invalid") 
        
    def set_dimred(self, dimred):
        self.dimred = dimred

    def generate_filename(self):
        if self.dimred.variant:
            prefix = self.criterion + '_' + self.dimred.variant
        else:
            prefix = self.criterion
        return self.dimred.generate_filename(self.dataset, self.get_folder(), prefix=prefix)
        
    def check_results(self):
        if not(self.eval_result.size):
            raise Exception("no evaluation results to plot. You first need to use eval()")

    def get_folder(self):
        if self.criterion == 'DIST':
            return 'dist_eval'
        else:
            return 'knn_eval'

    def eval(self, data_low=np.array([]), saveNload=True, verbose=True, end=False):
        if self.criterion == 'ATA':
            list_eval = np.array(self.eval_ata(verbose=verbose))
            self.eval_result = list_eval
            return list_eval
        
        if saveNload and not(end):
            file = self.load_eval()
            if file.size: 
                return file
            elif self.dimred.old_dimred: 
                return self.load_eval(new_param_range=True)
        
        if self.dimred.id == 'MDS' or self.dimred.parameter or end:
            if not(data_low.size):
                data_low = self.dimred.data_low
            if self.criterion == 'DIST':
                list_eval = self.eval_dist(data_low, verbose=verbose)
            else:
                list_eval = self.eval_kNN(data_low, verbose=verbose)
        else:
            list_eval = []
            param_range = self.dimred.get_range()
            list_data_low = self.dimred.data_low
            for i in range(len(list_data_low)):
                if verbose:
                    print(self.criterion + " evaluation of " + self.dimred.id + " on " + 
                        self.dataset.id + "\nwith Parameter=" + str(param_range[i]), end="")
                data_low = list_data_low[i]
                list_eval.append(self.eval(data_low=data_low, saveNload=saveNload, verbose=verbose, end=True))
        
        list_eval = np.array(list_eval)
        if saveNload and not(end):
            self.save_eval(list_eval)
        self.eval_result = np.array(list_eval)
        return list_eval

    def eval_dist(self, data_low, return_matrix=False, verbose=True):
        pair_dist1 = pairwise_distances(self.dataset.data_high, metric='euclidean')
        pair_dist2 = pairwise_distances(data_low, metric='euclidean')

        norm_pair_dist1 = np.divide(pair_dist1, np.max(pair_dist1))
        norm_pair_dist2 = np.divide(pair_dist2, np.max(pair_dist2))
        if return_matrix == 'beforediff':
            return abs(norm_pair_dist2)
        sub_norm_pair_dist = np.subtract(norm_pair_dist1, norm_pair_dist2)
        if return_matrix == 'diff':
            return abs(sub_norm_pair_dist)
        frob_norm = norm(sub_norm_pair_dist, ord='fro')
        if verbose:
            print(" and Frobenius Norm=" + str(frob_norm) + "\n")
        return frob_norm
    
    def eval_kNN(self, data_low, verbose=True):
        trustworthy = trustworthiness(self.dataset.data_high, data_low, n_neighbors=self.k)
        if verbose:
            print(" and Trustworthiness=" + str(trustworthy) + "\n")
        return trustworthy
    
    def eval_ata(self, verbose=True):
        dataset = self.dataset
        dimred = self.dimred
        evalcrit_dist = EvalCrit('DIST', dataset, dimred)
        evalcrit_1nn = EvalCrit('1NN', dataset, dimred)

        max_dist = 869
        min_dist = 243
        dist_result = evalcrit_dist.eval(verbose=False)
        nn_result = evalcrit_1nn.eval(verbose=False)

        if dimred.id != 'MDS':
            dist_result = [(-x+min_dist + max_dist) / max_dist for x in dist_result]
            ata_result = [statistics.harmonic_mean([x, y]) for x, y in list(zip(dist_result,nn_result))]
            X = np.array(dist_result)
            Y = np.array(nn_result)
            X, Y = np.meshgrid(X, Y)
            #Z = [statistics.harmonic_mean([x, y]) for x, y in list(zip(X,Y))]
            Z = np.array((X+Y)/2)
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.plot_surface(X, Y, Z, alpha=0.5)
            plt.show()
        else:
            dist_result = (-dist_result+min_dist + max_dist) / max_dist
            print(dist_result)
            print(nn_result)
            nn_result = int(nn_result)
            ata_result = statistics.harmonic_mean([dist_result, nn_result])
        if verbose:
            print(ata_result)

        return ata_result

    def plot_eval(self):
        self.check_results()
        list_eval = self.eval_result
        param_range = np.array(self.dimred.get_range())
        plt.figure(figsize=(10,8))
        #plt.scatter(param_range, list_eval, color='black')
        if not(self.dimred.id == 'MDS'):
            plt.plot(param_range, list_eval, color=self.dimred.plot_color)
            plt.xlabel("Parameter")
        else:
            plt.axhline(y = list_eval, color=self.dimred.plot_color)
            plt.xticks([])
        if self.criterion == 'DIST':
            plt.ylabel("Frobenius Norm")
        else:
            plt.ylabel("Trustworthiness")
        plt.title(self.criterion + " evaluation of " + self.dimred.id + " on " + self.dataset.id)
        plt.show()

    def plot_multiple_evals(evalcrits):
        criterion = evalcrits[0].criterion
        dataset_id = evalcrits[0].dataset.id
        plt.figure(figsize=(10,8))
        for eval in evalcrits:
            eval.check_results()
            list_eval = eval.eval_result
            param_range = np.array(eval.dimred.get_range())
            dimred_id = eval.dimred.id
            label = dimred_id
            plot_color = eval.dimred.plot_color
            if dimred_id == 'MDS':
                plt.axhline(y = list_eval, color=plot_color, label=dimred_id)
            else:
                variant = eval.dimred.variant
                if variant:
                    label = variant + dimred_id
                if variant == 'dense':
                    plt.scatter(param_range, list_eval, color=plot_color, label=label, s=100)
                else:
                    plt.plot(param_range, list_eval, color=plot_color, label=label)
        plt.xlabel("Parameter")
        if criterion == 'DIST':
            plt.ylabel("Frobenius Norm")
        else:
            plt.ylabel("Trustworthiness")
        plt.title(criterion + " evaluation on " + dataset_id)
        #plt.legend()
        plt.show()

    def plot_nn_evolution(evalcrits):
        dataset_id = evalcrits[0].dataset.id
        if dataset_id == 'COIL20':
            my_colors = ['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3',
                        '#c7eae5','#80cdc1','#35978f','#01665e','#003c30']
        else:
            my_colors = ['#130B01','#4C2C05','#744F1B','#A37E29','#DFAF2A',
                        '#E7C35F','#F5F5F5','#7ACDC0',
                        '#4EBCAB','#399385','#256A64','#01514A','#001410']
        plt.figure(figsize=(10,8))
        for i, eval in enumerate(evalcrits):
            eval.check_results()
            list_eval = eval.eval_result
            param_range = np.array(eval.dimred.get_range())
            if eval.dimred.id == 'MDS':
                plt.axhline(y = list_eval, color=my_colors[i], label=eval.criterion)
            else:
                plt.plot(param_range, list_eval, color=my_colors[i], label=eval.criterion)
        plt.xlabel("Parameter")
        plt.ylabel("Trustworthiness")
        plt.title("NN evaluation of " + eval.dimred.id + " on " + dataset_id)
        plt.legend()
        plt.show()
    
    def save_eval(self, list_eval):
        filename = self.generate_filename()
        if os.path.isfile(filename):
            raise Exception("File '" + filename + "' already exists")
        
        with open(filename, 'wb') as f:
            np.save(f, list_eval)
    
    def load_eval(self, new_param_range=False):
        if not(new_param_range):
            filename = self.generate_filename()
            if not(os.path.isfile(filename)):
                print("File '" + filename + "' does not exists")
                return np.array([])
            
            with open(filename, 'rb') as f:
                list_eval = np.load(f)
            print("File '" + filename + "' already exists and will be loaded..")
        else:
            old_dimred = self.dimred.old_dimred
            old_eval = EvalCrit(self.criterion, self.dataset, old_dimred)
            old_eval.eval()
            list_eval = [x[1] for x in zip(old_dimred.get_range(), old_eval.eval_result) if x[0] in self.dimred.get_range()]

        self.eval_result = np.array(list_eval)
        return list_eval

    def sort_print_eval(self, top, order=None):
        if not(self.dimred.id == 'MDS'):
            self.check_results()
            list_eval = self.eval_result
            param_range = np.array(self.dimred.get_range())
            df = pd.DataFrame(data={'parameter': param_range, 'value': list_eval})
            df.style.set_caption("hello")
            if not(order): order = self.order
            if order == 'ASC': ascending = True 
            elif order == 'DESC': ascending = False
            df = df.sort_values(by='value', ascending=ascending)
            df = df.head(top)
            df.index = list(range(1,top+1))
            print("\nTop " + str(top) + " " + self.criterion + " values in " + order + " order:")
            print(df)
        else:
            print(self.eval_result)

    def get_top_eval(self, to_return):
        if not(self.dimred.id == 'MDS'):
            self.check_results()
            list_eval = self.eval_result
            param_range = np.array(self.dimred.get_range())
            df = pd.DataFrame(data={'parameter': param_range, 'value': list_eval})
            place, to_return = to_return.split('.')
            if to_return == 'best':
                if self.criterion == 'DIST':
                    ascending = True
                else:
                    ascending = False
            elif to_return == 'worst':
                if self.criterion == 'DIST':
                    ascending = False
                else:
                    ascending = True
            df = df.sort_values(by='value', ascending=ascending)
            return df.iloc[int(place)-1]['parameter'], df.iloc[int(place)-1]['value']
        else:
            return None, self.eval_result

if __name__ == "__main__":
    ''' Time to calculate Mammoth embedding '''
    # param_range = [2,1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]
    # lle_values = [0.02,0.06,0.27,1.69,3.78,3.27,10.78,16.24,22.83,38.52,41.73]
    # tsne_values = [0.01,0.11,0.18,0.16,0.15,0.26,0.15,0.15,0.18,0.15,0.04]
    # plt.figure(figsize=(10,8))
    # plt.plot(param_range, lle_values, color='#d95f02', label='LLE')
    # plt.plot(param_range, tsne_values, color='#7570b3', label='TSNE')
    # plt.xlabel("Parameter")
    # plt.ylabel("Execution time in hours")
    # plt.title("Time to calculate Mammoth embedding")
    # plt.legend()
    # plt.show()

    ''' Time to calculate COIL-20 embedding '''
    param_range = [10,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400]
    lle_values = [0.08,0.22,0.67,1.28,2.1,2.95,4.37,5.83,7.22,9.03,10.88,12.98,13.35,18.18,20.68]
    tsne_values = [0.1,0.18,0.22,0.23,0.23,0.23,0.23,0.23,0.28,0.27,0.23,0.25,0.25,0.23,0.27]
    plt.figure(figsize=(10,8))
    plt.plot(param_range, lle_values, color='#d95f02', label='LLE')
    plt.plot(param_range, tsne_values, color='#7570b3', label='TSNE')
    plt.xlabel("Parameter")
    plt.ylabel("Execution time in minutes")
    plt.title("Time to calculate COIL20 embedding")
    plt.legend()
    plt.show()