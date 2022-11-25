# utils 

This package contains several functions for loading data, running experiments and visualizing results.

## Overview
| Module | Contents |
| --- | --- |
| [dataloading](dataloading.py) | functions for loading single patches (```load_patches```) or the whole FAUST and GALLOP training and testing datasets (```load_train_test_data_faust```, ```load_train_test_data_gallop```) and for converting a dataset into a PyTorch Geometric dataset (```get_pygeo_dataset```)|
| [experiment_runner](experiment_runner.py) | a class (```ExperimentRunner```) for running and saving experiments for multiple different models for multiple random seeds|
| [metrics](metrics.py) | a PyTorch implementation (```r2_score```) of the $R^{2}$-score|
| [model_trainer](model_trainer.py) | a class (```ModelTrainer```) for training and testing a given model on a given dataset |
| [plotting](plotting.py) | a function (```get_mesh_predictions```) to compute the nodewise mesh predictions of a given model as well as multiple functions for nicely visualising meshes or patches |
| [normalize](normalize.py) | a function (```normalize```) to normalize input meshes |
| [utils](utils.py) | functions to create directories and truncate colormaps |
| [srmesh_dataset_specnet](srmesh_dataset_specnet.py) | a class (```srmesh_dataset_specnet```) to evaluate the results |

