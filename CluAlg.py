from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from DataSet import DataSet
from DimRed import DimRed
import utils
import sys
import numpy as np
import os
from sklearn.metrics import pairwise_distances
import pandas as pd

class CluAlg():

    def kMeans(dataset, dimred=None, k=None, k_range=None):
        clualg = CluAlg(dataset, dimred)
        clualg.id = 'KMEANS'
        clualg.k = k
        clualg.k_range = k_range
        if k_range:
            clualg.min, clualg.max, clualg.step = k_range
            clualg.k_range = list(range(k_range[0], k_range[1], k_range[2]))
        return clualg

    def dbSCAN(dataset, dimred=None, eps=None, minpts=None, n=None, single=None):
        clualg = CluAlg(dataset, dimred)
        clualg.id = 'DBSCAN'
        if single:
            clualg.single_eps, clualg.single_minpts = single
            if dimred:
                data = dimred.data_low
            else:
                data = dataset.data_high
            data = pairwise_distances(data, metric='euclidean')
            min, max = int(np.min(data)), int(np.max(data))
            print(str(min) + '  ' + str(max))
        else:
            if eps:
                min, max, step = eps
                step = int(step) - 1
                if step < 1 : step = 1
            else:
                if dimred:
                    data = dimred.data_low
                    data = pairwise_distances(data, metric='euclidean')
                    min, max, step = np.min(data), np.max(data), np.max(data)/n
                    min += step
                    print(str(min) + '  ' + str(max) + '  ' + str(step))
                    clualg.min_eps, clualg.max_eps, clualg.step_eps = min, max, step
                    clualg.eps = list(np.arange(min, max, step))
                else:
                    data = dataset.data_high
                    data = pairwise_distances(data, metric='euclidean')
                    min, max, step = int(np.min(data)+1), int(np.max(data)), int(np.max(data)/n)
                    print(str(min) + '  ' + str(max) + '  ' + str(step))
                    clualg.min_eps, clualg.max_eps, clualg.step_eps = min, max, step
                    clualg.eps = list(range(min, max, step))

            if minpts:
                min, max, step = minpts
                step = int(step) - 1
                if step < 1 : step = 1
            else:
                min, max, step = 2, dataset.num_datapoints, int(dataset.num_datapoints/n)
            clualg.min_minpts, clualg.max_minpts, clualg.step_minpts = min, max, step
            clualg.minpts = list(range(min, max, step))
        return clualg

    def __init__(self, dataset, dimred=None):
        if dimred:
            self.high = False
            if dimred.id != 'MDS' and not(dimred.parameter):
                raise Exception('parameter for dimred must be set')
            self.single_eps = None
            self.single_minpts = None
        else:
            self.high = True
        self.dataset = dataset
        self.dimred = dimred
        self.ss = np.array([])
        self.sss = np.array([])
        self.ari = None
        self.ami = None
        self.eps = None
        self.minpts = None

    def transform(self, new_param):
        new_param = int(new_param)
        index = self.k_range.index(new_param)
        clustering = self.clusterings[index]
        clualg = CluAlg.kMeans(self.dataset, dimred=self.dimred, k=new_param)
        clualg.clustering = clustering
        return clualg

    def generate_filename(self, eval=None):
        filename = 'cluster//'
        if eval:
            filename += eval + '_'
        filename += self.id + '_'
        if self.high:
            filename += 'HIGH' + '_'
        else:
            filename += 'LOW' + '_'
            dimred_id = self.dimred.id
            filename += dimred_id + '_'
            if dimred_id == 'LLE' or dimred_id == 'TSNE':
                if self.dimred.parameter:
                    filename += str(self.dimred.parameter) + '_'
        filename += self.dataset.id + '_'
        if self.id == 'KMEANS':
            if self.k_range:
                filename += str(self.min) + '-' + str(self.max) + '-' + str(self.step)
            elif self.k:
                filename += str(self.k)
        elif self.id == 'DBSCAN':
            if self.single_eps:
                filename += 'eps(' + str(self.single_eps) + ')_'
                filename += 'mpts(' + str(self.single_minpts) + ')'
            else:
                filename += 'eps(' + str(self.min_eps) + '-' + str(self.max_eps) + '-' + str(self.step_eps) + ')_'
                filename += 'mpts(' + str(self.min_minpts) + '-' + str(self.max_minpts) + '-' + str(self.step_minpts) + ')'
        filename += '.npy'
        return filename

    def save_clustering(self, clustering):
        filename = self.generate_filename()
        if os.path.isfile(filename):
            print("File '" + filename + "' already exists")
        
        with open(filename, 'wb') as f:
            np.save(f, clustering)

    def load_clustering(self):
        filename = self.generate_filename()
        if not(os.path.isfile(filename)):
            print("File '" + filename + "' does not exists")
            file_content = np.array([])
        else:
            print("File '" + filename + "' already exists and will be loaded..")
            with open(filename, 'rb') as f:
                file_content = np.load(f)
        if self.id == 'KMEANS':
            if self.k:
                self.clustering = file_content
            elif self.k_range:
                self.clusterings = file_content
        elif self.id == 'DBSCAN':
            if self.single_eps:
                self.clustering = file_content
                values, counts = np.unique(file_content, return_counts=True)
                print(values)
                print(counts)
            else:
                self.clusterings = file_content
        return file_content

    def apply(self, saveNload=True, verbose=False):
        if saveNload:
            file_content = self.load_clustering()
            if file_content.size:
                return file_content
        
        clusterings = []
        if self.high:
            data = self.dataset.data_high
        else:
            data = self.dimred.data_low
        if self.id == 'KMEANS':
            if self.k:
                clusters = self.apply_kMeans(data, self.k)
            elif self.k_range:
                for k in self.k_range:
                    if verbose: print('calcing: k=' + str(k))
                    clusters = self.apply_kMeans(data, k)
                    clusterings.append(clusters)
        elif self.id == 'DBSCAN':
            if self.single_eps:
                clusters = self.apply_dbscan(data, self.single_eps, self.single_minpts)
            else:
                for eps in self.eps:
                    clustering = []
                    eps = np.round(eps, 7)
                    print('calcing: eps=' + str(eps))
                    for minpts in self.minpts:
                        if verbose:
                            print('calcing: eps=' + str(eps) + ', minpts=' + str(minpts))
                        clusters = self.apply_dbscan(data, eps, minpts)
                        clustering.append(clusters)
                    clusterings.append(clustering)
        
        if self.id == 'KMEANS':
            if self.k:
                self.clustering = clusters
                if saveNload:
                    self.save_clustering(clusters)
                return clusters
            elif self.k_range:
                self.clusterings = clusterings
                if saveNload:
                    self.save_clustering(clusterings)
                return clusterings
        elif self.id == 'DBSCAN':
            if self.single_eps:
                self.clustering = clusters
                values, counts = np.unique(clusters, return_counts=True)
                print(values)
                print(counts)
                if saveNload:
                    self.save_clustering(clusters)
                return clusters
            else:
                self.clusterings = clusterings
                if saveNload:
                    self.save_clustering(clusterings)
                return clusterings

    def apply_kMeans(self, data, k):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        clusters = kmeans.fit_predict(data)
        return clusters
    
    def apply_dbscan(self, data, eps, minpts):
        dbscan = DBSCAN(eps=eps, min_samples=minpts, n_jobs=-1)
        clusters = dbscan.fit_predict(data)
        return clusters
    
    def save_sss_eval(self, sss):
        filename = self.generate_filename('SC')
        if os.path.isfile(filename):
            print("File '" + filename + "' already exists")
        
        with open(filename, 'wb') as f:
            np.save(f, sss)

    def load_sss_eval(self):
        filename = self.generate_filename('SC')
        if not(os.path.isfile(filename)):
            print("File '" + filename + "' does not exists")
            sss = np.array([])
        else:
            print("File '" + filename + "' already exists and will be loaded..")
            with open(filename, 'rb') as f:
                sss = np.load(f)
        self.sss = sss
        return sss

    def ss_eval(self, saveNload=True):
        if saveNload:
            sss = self.load_sss_eval()
            if sss.size:
                return sss

        if self.high:
            data = self.dataset.data_high
        else:
            data = self.dimred.data_low
        
        if self.id == 'KMEANS':
            if self.k:
                ss = silhouette_score(data, self.clustering)
                self.ss = ss
                if not(self.high):
                    print(self.dimred.id + ' with P=' + str(self.dimred.parameter) + ' on ' + str(self.k) + ' clusters has SC=' + str(ss))
                else:
                    print('HIGH COIL20 on ' + str(self.k) + ' clusters has  SC=' + str(ss))
                return ss
            elif self.k_range:
                clusterings = self.clusterings
                sss = []
                for i, k in enumerate(self.k_range):
                    score = silhouette_score(data, clusterings[i])
                    if not(self.high):
                        print(self.dimred.id + ' ' + str(self.dimred.parameter) + ' ' + str(self.k) + ' sc=' + str(score))
                    else:
                        print('HIGH COIL20' + str(self.k) + ' sc=' + str(score))
                    sss.append(score)
                self.sss = np.array(sss)

                if saveNload:
                    self.save_sss_eval(sss)    
                return sss
        elif self.id == 'DBSCAN':
            if self.single_eps:
                try:
                    score = silhouette_score(data, self.clustering)
                    print(self.dimred.id + ' with P=' + str(self.dimred.parameter) + ' on eps=' +
                        str(self.single_eps) + ', minpts:' + str(self.single_minpts) + ' has SC=' + str(score))
                except:
                    print(self.clustering)
                return score
            else:
                clusterings = self.clusterings
                ssss = []
                for i, eps in enumerate(self.eps):
                    sss = []
                    for j, minpts in enumerate(self.minpts):
                        print(clusterings[i][j])
                        try:
                            score = silhouette_score(data, clusterings[i][j])
                        except:
                            score = -1
                        sss.append(score)
                        if not(self.high):
                            print(self.dimred.id + ' ' + str(self.dimred.parameter) + ' eps=' + str(eps) + ', minpts=' + str(minpts) + ' SC=' + str(score))
                        else:
                            print('HIGH COIL20 with eps=' + str(eps) + ', minpts=' + str(minpts) + ' SC=' + str(score))
                    ssss.append(sss)
                if saveNload:
                    self.save_sss_eval(ssss)    
                return ssss

    def ari_ami_eval(self, true_labels):
        pred_labels = self.clustering
        ari = adjusted_rand_score(true_labels, pred_labels)
        ami = adjusted_mutual_info_score(true_labels, pred_labels)
        print(self.dimred.id + ' with P=' + str(self.dimred.parameter) + ' on ' + str(self.k) + ' clusters has ARI=' + str(ari) + ' AMI=' + str(ami) + '\n')
        self.ari, self.ami = ari, ami
        return ari, ami

    def plot_eval(self):
        plt.figure(figsize=(10,8))
        title = ''
        if self.sss.size:
            title += 'SC '
            plt.plot(self.k_range, self.sss, color='#1b9e77', label='SC')
        if self.ari:
            title += 'ARI '
            plt.plot(self.k, self.ari, color='#d95f02', label='ARI')
        if self.ami:
            title += 'AMI '
            plt.plot(self.k, self.ami, color='#7570b3', label='AMI')
        if self.high:
            plt.ylabel("Silhouette coefficient")
            plt.title("SC evaluation of high dim. " + self.dataset.id + ', ' + self.id)
        else:
            plt.legend()
            plt.ylabel("Score value")
            plt.title(title + "evaluation of " + self.dimred.id + ' on ' + self.dataset.id + 
                      ' with Paramater=' + str(self.dimred.parameter) + ', ' + self.id)
        plt.xlabel("Number of clusters")    
        plt.grid()
        plt.show()

    def get_top_ss_eval(self):
        if self.ss.size:
            if self.ss.size == 1:
                print(str(self.k) + str(self.ss))
            else:
                ss = self.ss
                param_range = np.array(self.k)
                df = pd.DataFrame(data={'parameter': param_range, 'value': ss})
                df = df.sort_values(by='value', ascending=False)
                df = df.head(5)
                df.index = list(range(1,6))
                print("\nTop " + str(5) + " SC values:")
                print(df)
                return df
        elif self.sss.size:
            sss = self.sss
            eps = self.eps
            minpts = self.minpts

            best_i = 0
            best_j = 0
            best_s = -1
            secbest_i = 0
            secbest_j = 0
            secbest_s = -1
            for i, ss in enumerate(sss):
                for j, s in enumerate(ss):
                    if s > best_s:
                        secbest_i = best_i
                        secbest_j = best_j
                        secbest_s = best_s
                        best_i = i
                        best_j = j
                        best_s = s
            print('BEST: i=' + str(best_i) + ' j=' + str(best_j) + ' s=' + str(best_s))
            print('SECBEST: i=' + str(secbest_i) + ' j=' + str(secbest_j) + ' s=' + str(secbest_s))

    def plot_DBSCAN_heatmap(self, matrix):        
        fig, ax = plt.subplots(figsize=(12, 8))
        eps = self.eps
        minpts = self.minpts
        ax.set_title("HIGH COIL20 DBSCAN HEATMAP SC EVALUATION")
        ax.set_xlabel('minPts')
        ax.set_xticks(np.arange(len(minpts)), labels = minpts)
        ax.set_yticks(np.arange(len(eps)), labels = eps)
        ax.set_ylabel('eps')
        im = ax.imshow(matrix, cmap='hot_r')
        cbar = ax.figure.colorbar(im, ax = ax)
        cbar.ax.set_ylabel("Silhouette coefficient ", rotation = -90, va = "bottom")
        plt.show()       

def high(dataset, parameter):
    kmeans_high = CluAlg.kMeans(dataset, k_range=(2,100,1))
    kmeans_high.apply()
    kmeans_high = kmeans_high.transform(parameter)
    kmeans_high.ss_eval(saveNload=False)

def low_calc(dataset, dimred, parameter):
    print('CALC')
    kmeans_high = CluAlg.kMeans(dataset, k_range=(2,100,1))
    kmeans_high.apply()
    kmeans_high = kmeans_high.transform(parameter)
    kmeans_low = CluAlg.kMeans(dataset, dimred, k=parameter)
    kmeans_low.apply()
    kmeans_low.ss_eval(saveNload=False)
    kmeans_low.ari_ami_eval(kmeans_high.clustering)

def low_true(dataset, dimred, parameter, pred_labels):
    print('TRUE')
    kmeans_low = CluAlg.kMeans(dataset, dimred, k=parameter)
    kmeans_low.apply()
    kmeans_low.ss_eval(saveNload=False)
    kmeans_low.ari_ami_eval(pred_labels)

if __name__ == "__main__":
    ''' COIL"= '''
    dataset = DataSet('COIL20')
    dimred = DimRed('MDS')
    #dimred = DimRed('LLE', parameter_range=(2,1440,1))
    
    dimred.apply(dataset)
    #dimred = dimred.transform_method(new_param=835)
    #dataset.plot_data_low(dimred)
    # ''' HIGH COIL 20'''
    kmeans_high = CluAlg.kMeans(dataset, k_range=(2,100,1))
    kmeans_high.apply()
    kmeans_high.ss_eval()
    #kmeans_high.get_top_ss_eval()
    #kmeans_high.plot_eval()
    
    #low_true(dataset, dimred, 20, dataset.labels)
    low_calc(dataset, dimred, 20)
    # low_true(dataset, dimred, 72, dataset.image_nums)
    # low_calc(dataset, dimred, 72)

    
    # # dbscan_high = CluAlg.dbSCAN(dataset, eps=(0.1, 0.2, 0.1), minpts=(1,10,1))
    # # dbscan_high.apply()
    # ''' BEST: i=8 j=11 s=0.19405535024088622; SECBEST: i=5 j=1 s=0.14524586739048168 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, n=12)
    # ''' BEST -> BEST: i=7 j=7 s=0.19508087052502737 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, eps=(10732,13798,(13798-10732)/12) , minpts=(1202,1440,(1440-1202)/12))
    # ''' BEST -> BEST -> BEST: i=3 j=2 s=0.19508087052502737 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, eps=(12256,12764,(12764-12256)/12) , minpts=(1310,1346,(1346-1310)/12))
    # ''' BEST -> BEST -> BEST -> BEST: i=1 j=0 s=0.19508087052502737 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, eps=(12338,12420,(12420-12338)/12) , minpts=(1312,1316,(1316-1312)/12))
    # ''' BEST -> BEST -> BEST -> BEST -> BEST: i=2 j=0 s=0.19508087052502737: eps=12340, minpts=1312'''
    # #dbscan_high = CluAlg.dbSCAN(dataset, eps=(12338,12348,(12348-12338)/12) , minpts=(1312,1313,(1313-1312)/12))
    # ''' FINAL '''
    # # dbscan = CluAlg.dbSCAN(dataset, single=(12340, 1312))
    # # dbscan.apply()

    # ''' BEST: i=8 j=11 s=0.19405535024088622; SECBEST: i=5 j=1 s=0.14524586739048168 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, n=15)
    # ''' SECBEST -> BEST: i=11 j=0 s=0.19405535024088622 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, eps=(6133,9199,(9199-6133)/12) , minpts=(2,242,(242-2)/12))
    # ''' SECBEST -> BEST -> BEST: i=7 j=0 s=0.19405535024088622 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, eps=(8673,9181,(9181-8673)/12) , minpts=(2,21,(21-2)/12))
    # ''' SECBEST -> BEST -> BEST -> BEST: i=1 j=0 s=0.19405535024088622 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, eps=(8919,9001,(9001-8919)/12) , minpts=(2,3,(3-2)/12))
    # ''' SECBEST -> BEST -> BEST -> BEST -> BEST: i=1 j=0 s=0.19405535024088622: eps=8920, minpts=2 '''
    # #dbscan_high = CluAlg.dbSCAN(dataset, eps=(8919,8929,(8929-8919)/12) , minpts=(2,3,(3-2)/12))
    # ''' FINAL '''
    
    # # dbscan = CluAlg.dbSCAN(dataset, single=(8920, 2))
    # # dbscan.apply()

    # dbscan = CluAlg.dbSCAN(dataset, dimred, n=12)
    # dbscan.apply(saveNload=False, verbose=True)
    # ssss = dbscan.ss_eval(saveNload=False)
    # dbscan.plot_DBSCAN_heatmap(ssss)

    #ssss = dbscan.ss_eval(saveNload=True)
    
    #dbscan.get_top_ss_eval()
    #dbscan.plot_DBSCAN_heatmap(ssss)

    #dataset = DataSet('Mammoth')
    # dataset = DataSet('trans_Mammoth')
    #dimred = DimRed('MDS')
    #dimred = DimRed('LLE', parameter_range=(2,10000,1))
    #dimred = DimRed('LLE', parameter_range=(2,3334,1))
    # dimred = DimRed('TSNE', parameter_range=(2,10000,1))
    
    # dimred.apply(dataset)
    # dimred = dimred.transform_method(new_param=2)
    # #dataset.plot_data_low(dimred)
    
    # kmeans_high = CluAlg.kMeans(dataset, k_range=(2,3334,1))
    # kmeans_high.apply(verbose=True)
    # kmeans_high.ss_eval()
    # kmeans_high = kmeans_high.transform(2)
    # dataset.set_label_colors(kmeans_high.clustering)
    # dataset.plot_data_high()
    #kmeans_high.plot_eval()

    # low_calc(dataset, dimred, 2)

    
    # kmeans_low = CluAlg.kMeans(dataset, dimred, k=2)
    # kmeans_low.apply()
    # dataset.set_label_colors(kmeans_low.clustering)
    # dataset.plot_data_low(dimred)