import numpy as np
import json
import matplotlib.pyplot as plt
import os
import utils
from PIL import Image
from sklearn.metrics import pairwise_distances

class DataSet():
    
    def __init__(self, id, num_datapoints=None):
        self.id = id
        self.load_data_high()
        if id == 'COIL20':
            self.dim_low = 3
            self.set_label_colors(self.labels)
            if num_datapoints:
                print("INFO: num_datapoints will be ignored on COIL20")
            self.num_datapoints = 1440
        elif id == 'Mammoth' or id == 'trans_Mammoth':
            self.dim_low = 2
            with open('datasets//labels_mammoth_3d.npy', 'rb') as f:
                self.labels = np.load(f)
            self.set_label_colors(self.labels)
            if num_datapoints:
                self.data_high = self.data_high[:num_datapoints]
                self.num_datapoints = num_datapoints
            else:
                self.num_datapoints = 10000
        else:
            raise Exception("invalid dataset id")
        
    def set_label_colors(self, labels):
        value, _ = np.unique(labels, return_counts=True)
        unique_labels = len(value)
        if unique_labels > 20: raise Exception('number of labels must be <=20')
        my_colors_20 = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
                        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', 
                        '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
                        '#000075', '#a9a9a9']
        label_colors = []
        for l in labels:
            if self.id == 'COIL20':
                l = l-1
            label_colors.append(my_colors_20[l])
        self.label_colors = label_colors
    
    def transform(self, clustering, labels_to_trans, xyz, save=False):
        x, y, z = xyz
        indices = [i for i, c in enumerate(clustering) if c in labels_to_trans]
        for i in indices:
            self.data_high[i][0] += x
            self.data_high[i][1] += y
            self.data_high[i][2] += z
        if save:
            filename = 'datasets\\trans_mammoth_3d.npy'
            filename = 'datasets\\labels_mammoth_3d.npy'
            with open(filename, 'wb') as f:
                np.save(f, np.array(self.data_high))

    def plot_image_diff(self, to_plot):
        if to_plot == 'best':
            # img1: 481, img2: 482 with 0.018622213346064228
            image1Path = "datasets//coil-20-proc//obj7__49.png"
            image2Path = "datasets//coil-20-proc//obj7__50.png"
        elif to_plot == 'worst':
            # img1: 1156, img2: 769 with 1.0
            image1Path = "datasets//coil-20-proc//obj17__4.png"
            image2Path = "datasets//coil-20-proc//obj11__49.png"
        elif to_plot == 'same':
            # img1: 891, img2: 876 with 0.9177406572181258
            image1Path = "datasets//coil-20-proc//obj13__27.png"
            image2Path = "datasets//coil-20-proc//obj13__12.png"
        elif to_plot == 'diff':
            # img1: 881, img2: 1349 with 0.2224367341758462
            image1Path = "datasets//coil-20-proc//obj13__17.png"
            image2Path = "datasets//coil-20-proc//obj19__53.png"
        image1 = Image.open(image1Path)
        img1_array = np.asarray(image1)
        image2 = Image.open(image2Path)
        img2_array = np.asarray(image2)
        sub1 = abs(np.subtract(img2_array, img1_array))
        sub2 = abs(np.subtract(img1_array, img2_array))
        if sub1[0][0] < sub2[0][0]:
            image3 = sub1
        else:
            image3 = sub2
        image1.show()
        image2.show()
        #differenceImage = Image.fromarray(buffer3)
        #differenceImage.show()
        utils.plot_heatmap(image3, self, img=True)

    def get_pairwise_distances_high_dim(self):
        pair_dist = pairwise_distances(self.data_high, metric='euclidean')
        norm_pair_dist = np.divide(pair_dist, np.max(pair_dist))
        return abs(norm_pair_dist)

    def calc_img_info(self, matrix, to_calc):
        if not(self.id == 'COIL20'):
            raise Exception("no images to calculate info. You need to use COIL-20")
        img1, img2 = utils.get_matrix_value(matrix, to_calc)
        print("\nimg1: " + str(img1) + ", img2: " + str(img2) + " with " + str(matrix[img1][img2]))
        label1, label2 = self.labels[img1], self.labels[img2]
        img_num1, img_num2 = self.image_nums[img1], self.image_nums[img2]
        return label1, img_num1, label2, img_num2

    def open_coil20_img(self, label, img_num):
        path = 'datasets//coil-20-proc//obj' + \
            str(label) + '__' + str(img_num) + '.png'
        img = Image.open(path)
        print(path)
        img.show()

    def load_data_high(self):
        if self.id == 'COIL20':
            with open('datasets//coil-20-encoded.npy', 'rb') as f:
                self.data_high = np.load(f)
                self.labels = np.load(f)
                self.image_nums = np.load(f)
        else:
            self.labels = None
            if self.id == 'Mammoth':
                f = open('datasets//mammoth_3d.json')
                self.data_high = np.array(json.load(f))
                f.close()
            elif self.id == 'trans_Mammoth':
                with open('datasets//trans_mammoth_3d.npy', 'rb') as f:
                    self.data_high = np.load(f)

    def save_data_low(self, dimred, list_data_low):
        filename = dimred.generate_filename(self, 'data_low', prefix=dimred.variant)
        if os.path.isfile(filename):
            raise Exception("File '" + filename + "' already exists")
        else:
            with open(filename, 'wb') as f:
                np.save(f, list_data_low)

    def load_data_low(self, dimred):
        filename = dimred.generate_filename(self, 'data_low', prefix=dimred.variant)
        if not(os.path.isfile(filename)):
            raise Exception("File '" + filename + "' does not exists")
        else:
            with open(filename, 'rb') as f:
                list_data_low = np.load(f)
            return list_data_low

    def plot_data_high(self, colors=None, plot=True):
        if self.id == 'COIL20':
            raise Exception("High dimensional COIL20 dataset can\'t be plotted")
        else:
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            if colors.size:
                label_colors = colors
                cmap = 'hot_r'
                _, ax2 = plt.subplots(1, 1)
                map = ax2.imshow(np.stack([label_colors, label_colors]), cmap=cmap)
                fig.colorbar(map, ax=ax)
            else:
                label_colors = self.label_colors
                cmap = None
                values, counts = np.unique(self.label_colors, return_counts=True)
                print(values)
                print(counts)
            ax.scatter(self.data_high[:, 0], self.data_high[:, 1], self.data_high[:, 2], s=1, c=label_colors, cmap=cmap)
            if plot: plt.show()

    def plot_data_low(self, dimred, colors=None, plot=True):
        if dimred.parameter_range:
            raise Exception("result of parameter range can\'t be plotted")
            
        fig = plt.figure(figsize=(10,8))
        data_low = dimred.data_low
        if colors:
            label_colors = colors
            cmap = 'hot_r'
        else:
            label_colors = self.label_colors
            cmap = None
        if self.id == 'COIL20':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_low[:, 0], data_low[:, 1], data_low[:, 2], c=label_colors, cmap=cmap)
        else:
            ax = fig.add_subplot(111)
            ax.scatter(data_low[:, 0], data_low[:, 1], s=1, c=label_colors, cmap=cmap)
        
        ax.set_title(dimred.id + " on " + self.id + " with Parameter=" + str(dimred.parameter))
        if plot: plt.show()

if __name__ == "__main__":
    pass