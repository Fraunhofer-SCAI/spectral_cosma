from .model_trainer import ModelTrainer
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
except: 
    sns = None
    
    
class ExperimentRunner:
    """
    **********************
    ** ExperimentRunner **
    **********************
    
        Class for running experiments with different models and seeds.
        
        -----
        args:
        -----
            - model_builders:      dict;   dictionary containing functions for initializing the models that will be considered in the experiment
            - optimizer_builders:  dict;   dictionary containing functions for initializing the optimizers that will be considered in the experiment
            - loss_func_builders:  dict;   dictionary containing functions for initializing the loss functions that will be considered in the experiment
            - save_directory:      string; the directory in which the trained model and the training stats will be saved
            - device:              string; the device used by pytorch
    """
    
    def __init__(self, 
                 model_builders, 
                 optimizer_builders,
                 loss_func_builders,
                 dataset_builders,
                 save_directory = 'experiments/test_experiment/',
                 clip_grad_norm=False, 
                 grad_max_norm = 0.1,
                 device = None):
        self.model_builders = model_builders
        self.optimizer_builders = optimizer_builders
        self.loss_func_builders = loss_func_builders
        self.dataset_builders = dataset_builders
        self.names = self.model_builders.keys()
        self.save_directory = save_directory
        self.clip_grad_norm = clip_grad_norm
        self.grad_max_norm = grad_max_norm
        self.device = device
        self.dataset = None
    
    def get_trained_model(self, seed, name, best_validation = True):
        """Return the trained model corresponding to the given name and seed"""
        model_name = 'saved_model' if not best_validation else 'best_validation_model'
        path = self.save_directory + name + '/seed_{0}/{1}.pth'.format(seed, model_name)
        model = self.model_builders[name]()
        model.load_state_dict(torch.load(path, map_location = self.device))
        if self.device is not None:
            model = model.to(self.device)
        return model
    
    def get_dataset(self, name, seed):
        """Return train and test dataset shuffled according to the given seed"""
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_dataset, test_dataset = self.dataset_builders[name]() 
        return train_dataset, test_dataset
    
    def get_model_trainer(self, name, seed, logfile=None):
        """Return an instance of ModelTrainer corresponding to the given name and seed"""
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # get model trainer arguments
        model = self.model_builders[name]()
        if logfile is not None:
            params = list(model.parameters())
            #print(len(params))
            trainable_weights = 0
            for ii in range(len(params)):
                #print(params[ii].size())
                trainable_weights += (np.prod(params[ii].size()))
            print(trainable_weights,'trainable weights')
            logfile.write('{} trainable weights\n'.format(trainable_weights))
        # move model to device
        model = model if self.device is None else model.to(self.device)
        optimizer = self.optimizer_builders[name](model)
        loss_func = self.loss_func_builders[name]()
        save_directory = self.save_directory + name + '/seed_{0}/'.format(seed)
        # define model trainer
        trainer = ModelTrainer(model = model, 
                               optimizer = optimizer,
                               loss_func = loss_func,
                               save_dir = save_directory,
                               clip_grad_norm = self.clip_grad_norm, 
                               grad_max_norm = self.grad_max_norm,
                               device = self.device)
        return trainer
            
    def plot_learning_curves(self, seeds, test_interval, filename='avg_learning_curves'):
        """Plot the learning curves of the experiment"""
        training_stats = self.get_avg_stats(seeds)
        titles = [['Train-Loss', 'Validation-Loss'], ['Train-R^2-Score', 'Validation-R^2-Score']]
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Learning Curves') 
        fig.tight_layout() #pad=4)
        for i in range(2):
            for j in range(2):
                title = titles[i][j]
                axs[i, j].set_title(title)
                axs[i, j].set_xlabel('Epoch')
                for name in training_stats.keys():
                    if 'Validation' in title:
                        axs[i, j].plot(training_stats[name][title][::test_interval], label = name)
                    else:
                        axs[i, j].plot(training_stats[name][title], label = name)
                if 'Score' in title:
                    axs[i, j].set_ylim(0.0, 1.01)
                else:
                    axs[i, j].set_ylim(0, 10 * training_stats[name]['Validation-Loss'].min())
        handles, labels = axs[0, 0].get_legend_handles_labels()
        lgd = fig.legend(handles, labels, loc = 'right')
        plt.savefig(self.save_directory + '{}.png'.format(filename), dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
        
    def get_avg_stats(self, seeds):
        """Return the training statistics averaged over the given seeds"""
        training_stats = {}
        for name in self.names:
            training_stats[name] = pd.read_csv(self.save_directory + name + '/seed_{0}/training_stats.csv'.format(seeds[0]), dtype = np.single)
            for seed in seeds[1:]:
                training_stats[name] += pd.read_csv(self.save_directory + name + '/seed_{0}/training_stats.csv'.format(seed), dtype = np.single)
            training_stats[name] /= len(seeds)   
        return training_stats
    
    def get_param_num(self, model_name):
        """Return the number of parameters of the model corresponding to model_name"""
        model = self.model_builders[model_name]()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return int(num_params)
    
    def get_param_col(self):
        """Add a column with the number of parameters to the final training statistics"""
        param_nums = {}
        for name in self.names:
            param_nums[name] = {'Parameters': self.get_param_num(name)}
        return pd.DataFrame(param_nums)
    
    def plot_final_stats(self, seeds):
        """Plot the final training statistics"""
        plt.figure()
        avg_stats = self.get_avg_stats(seeds)
        final_stats = {}
        for name in self.names:
            final_stats[name] = avg_stats[name].iloc[-1]
        df = pd.concat([pd.DataFrame(final_stats), self.get_param_col()]).T
        scaled_df = (df - df.min(axis=0))/(df.max(axis=0) - df.min(axis=0))
        scaled_df['Train-Loss'] = - scaled_df['Train-Loss'] + 1
        scaled_df['Runtimes'] = - scaled_df['Runtimes'] + 1
        scaled_df['Parameters'] = - scaled_df['Parameters'] + 1
        scaled_df['Validation-Loss'] = - scaled_df['Validation-Loss'] + 1
        
        df['Parameters'] = df['Parameters'].astype(int)
        df['Runtimes'] = df['Runtimes'].astype(int)
        df[['Train-Loss', 'Validation-Loss']] = df[['Train-Loss', 'Validation-Loss']].applymap('{:,.2e}'.format)
        df[['Train-R^2-Score', 'Validation-R^2-Score']] = df[['Train-R^2-Score', 'Validation-R^2-Score']].applymap('{:,.4f}'.format)
        df[['Runtimes', 'Parameters']] = df[['Runtimes', 'Parameters']].applymap('{:d}'.format)
        
        hm = sns.heatmap(scaled_df, annot=df, linewidths=.5, cmap="RdYlGn", cbar = False, fmt = '')
        hm.set_xticklabels(hm.get_xticklabels(), rotation=45) 
        plt.title('Final Training Statistics Comparison')
        plt.savefig(self.save_directory + 'final_stats.png', dpi = 600, bbox_inches='tight')      
        
    def run_experiment(self, seeds = [1, 2, 3, 4, 5], num_epochs = 100, test_interval = 10, save_interval = 10, plot_results = False, logfile = None, writer = None):
        """Run an experiment
           -----
           args:
           -----
               - seeds:         list; the random seeds used for the experiment
               - num_epochs:    int;  the number of epochs to train the models
               - test_interval: int;  a test on the validation data is performed every test_interval-th epoch
               - save_interval: int;  the models and training statistics are saved every save_interval-th epoch
        """
        for name in self.names:
            for seed in seeds:
                # get trainer and datasets for given name and seed
                trainer = self.get_model_trainer(name, seed, logfile=logfile)
                train_dataset, test_dataset = self.get_dataset(name, seed)
                # train the model with the given name and seed
                print('\n\n Training Model {0} for seed {1}:\n'.format(name, seed))
                if logfile is not None:
                    logfile.write('\n\n Training Model {0} for seed {1}:\n\n'.format(name, seed))
                trainer.train(train_dataset = train_dataset, 
                              test_dataset = test_dataset, 
                              num_epochs = num_epochs, 
                              test_interval = test_interval, 
                              save_interval = save_interval,
                              use_edge_index = False if not ('GCN' in name or 'ChebNet' in name) else True,
                              logfile = logfile,
                              writer = writer)
        if plot_results:
            self.plot_learning_curves(seeds, test_interval)
            self.plot_final_stats(seeds)
                