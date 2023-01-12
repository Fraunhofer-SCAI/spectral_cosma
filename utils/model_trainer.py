import torch
import numpy as np
import time
import pandas as pd
from .metrics import r2_score
import os


class ModelTrainer:
    """
    ******************
    ** ModelTrainer **
    ******************
    
        Class for training and testing a model with detailed logs.
        
        -----
        args:
        -----
            - model:      object;   the model that will be trained
            - optimizer:  object;   the optimizer used to update the model
            - loss_func:  callable; the loss function used to update the model
            - save_dir:   string;   the directory in which the trained model and the training stats will be saved
            - device:     string;   the device used by pytorch
    """
    
    def __init__(self, model, optimizer, loss_func, save_dir = None, 
                 clip_grad_norm = False, grad_max_norm = 0.1, 
                 device = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.save_dir = save_dir
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.device = device
        self.best_val_loss = np.inf
        self.clip_grad_norm = clip_grad_norm
        self.grad_max_norm = grad_max_norm
    
    """
    Does moving the zero_grad() in the main loop after the forward pass solve the issue (just before the loss.backward())?

    Also using .backward() in your forward pass is quite dangerous, you don't know how much it will impact as it will backprop on anything that impacted x which does not seem to be what you want here right? Could you use autograd.grad() to get only the gradient you want?
    """    
    def update(self):
        """Update the model weights by performing one whole batch update on the train_dataset"""
        self.model.train()
        total_norm, mean_loss, mean_reg, mean_score = 0, 0, 0, 0
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        for data in self.train_dataset:
            #self.optimizer.zero_grad() 
            data = data if self.device is None else data.to(self.device)
            out = self.model(data.x, data.edge_index) if self.use_edge_index else self.model(data.x)
            loss_mse, reg = self.loss_func(out, data.y, loss_weights = data.loss_weights)
            loss = loss_mse  # omit regularization 
            # Zero the gradients before running the backward pass.
            self.optimizer.zero_grad()
            loss.backward()

            norm = 0
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                norm += param_norm.item() ** 2
            norm = norm ** 0.5
            
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)

            self.optimizer.step()
            
            total_norm += norm
            mean_loss += loss.item()
            mean_reg += reg.item()
            mean_score += r2_score(out, data.y).item() #r2_score(out.detach().numpy(),data.y.detach().numpy()) 
        return total_norm/len(self.train_dataset), mean_loss/len(self.train_dataset), mean_reg/len(self.train_dataset), mean_score/len(self.train_dataset)
        
    def validate(self):
        """Validate the model on the whole test_dataset"""
        self.model.eval()
        loss, reg, score = 0, 0, 0
        for data in self.test_dataset:
            data = data if self.device is None else data.to(self.device)
            out = self.model(data.x, data.edge_index) if self.use_edge_index else self.model(data.x)
            new_loss, new_reg = self.loss_func(out, data.y, loss_weights = data.loss_weights)
            loss += new_loss.item()
            reg += new_reg.item()
            score += r2_score(out,data.y).item() #r2_score(out.detach().numpy(),data.y.detach().numpy())  
        return loss/len(self.test_dataset), reg/len(self.test_dataset), score/len(self.test_dataset)    
    
    def save_model(self, val_loss = None):
        """Save the current model weights"""
        if val_loss is None:
            torch.save(self.model.state_dict(), self.save_dir + 'saved_model.pth')
        else:
            if val_loss <= self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_dir + 'best_validation_model.pth')
        
    def save_stats(self):
        """Save the current training stats to csv"""
        if len(self.val_scores):
            rep = len(self.scores) / len(self.val_scores)
        else: 
            rep = len(self.scores)
        stats = {'Train-Loss': np.array(self.losses),
                 'Train-Regularization': np.array(self.losses_reg),
                 'Gradient-Norm': np.array(self.gnorms),
                 'Train-R^2-Score': np.array(self.scores),
                 'Validation-Loss': np.repeat(self.val_losses, rep),
                 'Validation-Regularization': np.repeat(self.val_losses_reg, rep),
                 'Validation-R^2-Score': np.repeat(self.val_scores, rep),
                 'Runtimes': np.repeat(self.runtimes, rep)}
        pd.DataFrame(stats).to_csv(self.save_dir + 'training_stats.csv', index = False)
        
    
    def train(self, train_dataset, test_dataset, 
              num_epochs = 1000, test_interval = 100, save_interval = 100,
              use_edge_index = False, logfile = None, writer = None):
        """Train the model on a given train_dataset
               
           -----
           args:
           -----
               - train_dataset:  list of objects; the dataset the model will be trained on
               - test_dataset:   list of objects; the dataset the model will be validated on
               - num_epochs:     int;             the number of epochs to train the model
               - test_interval:  int;             the model will be validated every test_interval-th epoch
               - save_interval:  int;             the model and training stats will be saved every save_interval-th epoch (has to be multiple of test_interval)
               - use_edge_index: bool;            sturn true if model corresponds to a Pytorch Geometric Graph Neural Network
        """
        self.start = time.time()
        # init train and test dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.use_edge_index = use_edge_index
        
        # init training stats
        self.losses = []
        self.losses_reg = []
        self.gnorms = []
        self.scores = []
        self.val_losses = []
        self.val_losses_reg = []
        self.val_scores = []
        self.runtimes = []
        self.runtimeswo = []
        
        # training loop
        for epoch in range(1, num_epochs + 1):
            # update the model in one epoch
            gnorm, loss, reg, score = self.update()
            self.losses.append(loss)
            self.losses_reg.append(reg)
            self.gnorms.append(gnorm)
            self.scores.append(score)
            # validate the model every test_interval-th epoch
            if epoch % test_interval == 0:
                self.runtimeswo.append(time.time() - self.start)
                val_loss, val_reg, val_score = self.validate()
                self.val_losses.append(val_loss)
                self.val_losses_reg.append(val_reg)
                self.val_scores.append(val_score)
                self.runtimes.append(time.time() - self.start)
                runtime = self.runtimes[-1] - self.runtimes[-2] if epoch // test_interval > 1 else self.runtimes[-1] 
                runtimewo = self.runtimeswo[-1] - self.runtimeswo[-2] if epoch // test_interval > 1 else self.runtimeswo[-1] 
                logtext = f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Gradient Norm: {gnorm:.3f}, R^2 Score: {score:.5f}, Validation-Loss: {val_loss:.6f}, Validation R^2 Score: {val_score:.5f}, Runtime: {runtime:.2f}'
                
                print(logtext)
                if logfile is not None: 
                    logfile.write( logtext+'\n')
                self.save_model(val_loss)
            # save the current model weights and training stats every save_interval-th epoch
            if epoch % save_interval == 0 and self.save_dir is not None:
                self.save_model()
                self.save_stats()
        
            if writer is not None:
                # adding running loss to tensorboard
                writer.add_scalar("Loss/train", loss, epoch)
                writer.add_scalar("GradientNorm", gnorm, epoch)
                if epoch % test_interval == 0:
                    writer.add_scalar("Loss/val", val_loss, epoch)
        
        print('Runtime: ', self.runtimes[-1])
        if logfile is not None: 
            logfile.write('Runtime: {}\n'.format(self.runtimes[-1]))
        
        return {'Loss': np.array(self.losses),
                'Regularization': np.array(self.losses_reg),
                'Gradient Norm': np.array(self.gnorms),
                'R^2 Score': np.array(self.scores),  
                'Validation-Loss': np.array(self.val_losses), 
                'Validation-Regularization': np.array(self.val_losses_reg), 
                'Validation R^2 Score': np.array(self.val_scores),
                'Runtimes': np.array(self.runtimes)}