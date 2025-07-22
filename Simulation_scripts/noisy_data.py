################################################################################  
#           Simulation with noise only present in the test set.                # 
#                                                                              #
#   We vary the total sample size and amount of noise, then train the model    #
#   for each combination.                                                      #
#   Model performance is evaluated using test accuracy and cross-entropy.      #
#                                                                              #
################################################################################

import numpy as np 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from timeit import default_timer as timer 
import joblib
from joblib import Parallel, delayed
import modular.samples_setup as cs
from modular import engine
from modular import extra_functions as ef
from modular import model_builder

SEED = 42
NREPS = 50
BATCH_SIZE = 32
EPOCHS = 10

n_samples = [[1000]*2,[2500]*2, [5000]*2,[10000]*2]
vars = np.array([0.05,0.1,0.2,0.35])

# Save accuracy and C-E
save_results = [
    (
        np.zeros((len(n_samples),len(vars),NREPS)),
        np.zeros((len(n_samples),len(vars),NREPS))
    )
    for _ in range(4)
]

loss_fn = nn.CrossEntropyLoss()

# Set the seed and start the timer
torch.manual_seed(SEED)
train_time_start_on_cpu = timer()


def run_model(test_set,test_label,train_set, train_label):
    
    # Create tensor
    X_test = torch.from_numpy(test_set).type(torch.float)
    X_train = torch.from_numpy(train_set).type(torch.float)

    y_train = torch.from_numpy(train_label).type(torch.long)
    y_test = torch.from_numpy(test_label).type(torch.long)
    
    ## Add channel at dimension 1 
    X_train = X_train.unsqueeze(1)  
    X_test = X_test.unsqueeze(1)  
            
    train_dataset = torch.utils.data.TensorDataset(X_train,y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test,y_test)
            
    # Create data loader 
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
        ) 
                                
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
        )
    
    # Start model and optimizer
    model = model_builder.TVGG(
        input_shape=1,  
        hidden_units=10, 
        output_shape=2
        ) 
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    # Train 
    output = engine.train_test_loop(
        model,
        train_dataloader,
        test_dataloader, 
        optimizer, 
        loss_fn,
        epochs=EPOCHS,
        print_b=False
        )
    return (output)

def all_process(n,v,k,n_sample,var):
    
    # Simulate data
    output = cs.generate_sample(n=n_sample, noise_prop=0,var=0)
    images, labels= (output['images'], output['labels'])
    
    # Split test and train
    n_test = int(sum(n_sample)*0.2)
    test_index = np.arange(n_test)
    train_index = np.arange(n_test, sum(n_sample))

    images_test = images[test_index]
    images_train = images[train_index]

    label_test = labels[test_index]
    label_train = labels[train_index]
            
    # 0. Without noise
    output = run_model(images_test,label_test,images_train,label_train)

    acc_e0 = output['test_acc'][EPOCHS-1]
    ce_e0 = output['test_ce'][EPOCHS-1]
                    
    # 1. Adding noise only to the test set
    images_test_e1 = images_test.copy()

    for i in range(len(images_test_e1)):
            images_test_e1[i] = ef.add_gaussian_noise(
                images_test_e1[i],
                var=var
                )
    
    output = run_model(images_test_e1,label_test,images_train,label_train)

    acc_e1 = output['test_acc'][EPOCHS-1]
    ce_e1 = output['test_ce'][EPOCHS-1]
    
    # 2. Adding noise only to the train set
    images_train_e2 = images_train.copy()

    for i in range(len(images_train_e2)):
            images_train_e2[i] = ef.add_gaussian_noise(
                images_train_e2[i],
                var=var
                )
    
    output = run_model(images_test,label_test,images_train_e2,label_train)

    acc_e2 = output['test_acc'][EPOCHS-1]
    ce_e2 = output['test_ce'][EPOCHS-1]
    
    # 2. Adding noise only to both 
    images_train_e3 = images_train.copy()
    images_test_e3 = images_test.copy()

    for i in range(len(images_train_e3)):
            images_train_e3[i] = ef.add_gaussian_noise(
                images_train_e3[i],
                var=var
                )
    for i in range(len(images_test_e3)):
        images_test_e3[i] = ef.add_gaussian_noise(
            images_test_e3[i],
            var=var
            )
        
    output = run_model(images_test_e3,label_test,images_train_e3,label_train)

    acc_e3 = output['test_acc'][EPOCHS-1]
    ce_e3 = output['test_ce'][EPOCHS-1]
    
    all_outputs = [
        (acc_e0,ce_e0),
        (acc_e1,ce_e1),
        (acc_e2,ce_e2),
        (acc_e3,ce_e3)
    ]
    
    return(n,v,k,all_outputs)
    
# Parallelize the outermost loop
outputs = Parallel(n_jobs=100)(
    delayed(all_process)(
        n, v, k, n_sample, var
        )
        for v, var in enumerate(vars)
        for n, n_sample in enumerate(n_samples)
        for k in range(NREPS)
    )

for n,v,k, all_results in outputs:
    for idx,(acc,ce) in enumerate(all_results):        
        save_results[idx][0][n,v,k] = acc
        save_results[idx][1][n,v,k] = ce
                
# Save Environment
env_vars = {
    'n_samples' : n_samples,
    'vars': vars,
    'save_results': save_results,
    'NREPS': NREPS,
    'EPOCHS': EPOCHS,
    'BATCH_SIZE': BATCH_SIZE
    }       
        
# Save variables using joblib 
joblib.dump(env_vars, '/u/ruizsuar/InformedML-CV/Environments/ND_Dec2.pkl')
                                    