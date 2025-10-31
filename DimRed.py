from sklearn.manifold import MDS, LocallyLinearEmbedding, TSNE
import numpy as np
import os
import time


class DimRed():
    
    def __init__(self, id, parameter=None, parameter_range=None, temp=False, variant='', parameters=None):
        if id == 'MDS':
            if parameter:
                print("INFO: parameter will be ignored for MDS")
            if parameter_range:
                print("INFO: parameter_range will be ignored for MDS")
        elif id == 'LLE' or id == 'TSNE':
            if not(parameter or parameter_range or parameters):
                raise Exception("either parameter or parameter_range or parameters must be set")

        self.id = id
        self.parameter = parameter
        self.parameter_range = parameter_range
        self.parameters = parameters
        self.temp = temp
        self.old_dimred = None
        self.variant = variant
        if id == 'MDS':
            self.plot_color = '#1b9e77'
        elif id == 'LLE':
            if variant == 'dense':
                self.plot_color = 'black'
            else:
                self.plot_color = '#d95f02'
        elif id == 'TSNE':
            if variant == 'random':
                self.plot_color = 'black'
            else:
                self.plot_color = '#7570b3'

    def set_data_low(self, data_low):
        self.data_low = data_low

    def set_old_dimred(self, old_dimred):
        self.old_dimred = old_dimred

    def transform_method(self, save_old=True, new_param=None, new_param_range=None):
        if self.id == 'MDS':
            method = self
        else:
            if new_param:
                print("test")
                new_param = int(new_param)
                index = self.get_range().index(new_param)
                data_low = self.data_low[index]
                method = DimRed(self.id, parameter=new_param)
            elif new_param_range:
                old_param_range = self.get_range()
                new_min, new_max, new_step = new_param_range
                old_min, old_max, old_step = self.get_min_max_step()
                if new_step % old_step:
                    raise Exception("new step size must be multiple of the old step size")
                if not(new_min in old_param_range or new_min<0):
                    raise Exception("new minimum must be in the old parameter range or <0")
                if not(old_max-1 in old_param_range) or new_max < new_min:
                    raise Exception("new maximum-1 must be in the old parameter range and greater then the new minimum")
                data_low = [x[1] for x in zip(old_param_range, self.data_low) if x[0] in list(range(new_min, new_max, new_step))]
                method = DimRed(self.id, parameter_range=new_param_range, variant=self.variant)
            else:
                raise Exception("either new_param or new_param_range must be set")
        
            method.set_data_low(data_low)
            if save_old:
                method.set_old_dimred(self) 
        return method

    def generate_filename(self, dataset, folder, prefix=''):
        filename = folder + '//'
        if prefix:
            filename += prefix + '_'
        filename += self.id + '_' + dataset.id
        if self.id == 'LLE' or self.id == 'TSNE':
            if self.parameter:
                filename += '__' + str(self.parameter)
            elif self.parameters:
                filename += '__' + '_'.join(map(str,self.parameters))
            else:
                min, max, step = self.get_min_max_step()
                filename += '__' + str(min) + "-" + str(max) + "-" + str(step)
        suffix = '.npy'
        return filename + suffix

    def get_min_max_step(self):
        if not(self.parameter_range):
            raise Exception("no parameter range defiend")
        else:
            return self.parameter_range[0], self.parameter_range[1], self.parameter_range[2]
    
    def get_range(self):
        if self.id == 'MDS':
            return 1
        elif self.parameter:
            return self.parameter
        elif self.parameters:
            return self.parameters
        else:
            min, max, step = self.get_min_max_step()
            return range(min, max, step)

    def apply(self, dataset, saveNload=True, verbose=True):
        if saveNload:
            filename = self.generate_filename(dataset, 'data_low', prefix=self.variant)
            if os.path.isfile(filename):
                print("File '" + filename + "' already exists and will be loaded..")
                list_data_low = np.array(dataset.load_data_low(self))
                self.set_data_low(list_data_low)
                return list_data_low
        
        if self.parameter or self.id == 'MDS':
            list_data_low = self.apply_single(dataset, verbose=verbose)
        else:
            list_data_low = []
            start_entire = time.time()
            for parameter in self.get_range():
                temp_dimred = DimRed(self.id, parameter=parameter, temp=True)
                start = time.time()
                list_data_low.append(temp_dimred.apply(dataset))
                end = time.time()
                if verbose:
                    print("This calculation took: " + str(end-start) + " seconds\n")
                # if end-start > 60:
                #     self.parameter_range = list(self.parameter_range)
                #     self.parameter_range[1] = parameter
                #     self.parameter_range = tuple(self.parameter_range)
                #     break
            end_entire = time.time()
            if verbose:
                print("This ENTIRE calculation took: " + str(end_entire-start_entire) + " seconds\n")

        list_data_low = np.array(list_data_low)

        if saveNload and not(self.temp):
            dataset.save_data_low(self, list_data_low)

        self.set_data_low(list_data_low)
        return list_data_low

    def apply_single(self, dataset, parameter=None, verbose=True):
        if not(parameter):
            parameter = self.parameter
        if not(parameter) and not(self.id == 'MDS'):
            raise Exception("parameter must be set")
        
        if verbose:
            print(self.id + " on " + dataset.id + " with Parameter=" + str(parameter) + " processing..")
        
        if self.id == 'MDS':
            embedding = MDS(n_components=dataset.dim_low, n_jobs=-1)
            data_low = embedding.fit_transform(dataset.data_high)
        elif self.id == 'LLE':
            if self.variant == 'dense':
                embedding = LocallyLinearEmbedding(n_components=dataset.dim_low, n_neighbors=parameter, n_jobs=-1, eigen_solver='dense')
                data_low = embedding.fit_transform(dataset.data_high)
            else:
                try:
                    embedding = LocallyLinearEmbedding(n_components=dataset.dim_low, n_neighbors=parameter, n_jobs=-1, eigen_solver='arpack')
                    data_low = embedding.fit_transform(dataset.data_high)
                except:
                    embedding = LocallyLinearEmbedding(n_components=dataset.dim_low, n_neighbors=parameter, n_jobs=-1, eigen_solver='dense')
                    data_low = embedding.fit_transform(dataset.data_high)
        elif self.id == 'TSNE':
            if self.variant == 'random':
                embedding = TSNE(n_components=dataset.dim_low, perplexity=parameter, n_jobs=-1, init='random')
            else:
                embedding = TSNE(n_components=dataset.dim_low, perplexity=parameter, n_jobs=-1, init='pca')
            data_low = embedding.fit_transform(dataset.data_high)
        
        return data_low
    
if __name__ == "__main__":
    pass