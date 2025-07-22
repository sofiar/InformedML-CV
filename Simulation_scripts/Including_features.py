import torch 
import numpy as np 
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
from joblib import Parallel, delayed
import modular.samples_setup as cs
from modular import engine
from modular import model_builder


################################################################################
#                          Functions definitions                               #
################################################################################

# function to loop over
def process_combination(n, nsample, k, data, size_c, size_s,nscenarios,m):
    # Get data
    output = data[n][k]
    images, labels = (output['images'], output['labels'])

    # Split test and train
    n_test = int(sum(nsample)*0.2)
    test_index = np.arange(n_test)
    train_index = np.arange(n_test, sum(nsample))

    images_test = images[test_index]
    images_train = images[train_index]

    label_test = labels[test_index]
    label_train = labels[train_index]

    # keep indices
    indx_s = np.where(labels==1)
    indx_c = np.where(labels==0)        
    
    outputs = []
    for i in range(nscenarios):

        # Get extra feature
        feature_c = size_c[n][i,k,]
        feature_s = size_s[n][i,k,]

        extra_feature = np.zeros(len(labels))
        extra_feature[indx_c] = feature_c
        extra_feature[indx_s] = feature_s 

        # Repeat each value as a 28x28 matrix to create channel
        ef_arr = np.tile(extra_feature[:, np.newaxis, np.newaxis], (1, 28, 28))
        ef_test = ef_arr[test_index]
        ef_train = ef_arr[train_index]
            
        # Create tensors
        X_test = torch.from_numpy(images_test).type(torch.float)
        X_train = torch.from_numpy(images_train).type(torch.float)

        F_test = torch.from_numpy(ef_test).type(torch.float)
        F_train = torch.from_numpy(ef_train).type(torch.float)

        y_train = torch.from_numpy(label_train).type(torch.long)
        y_test = torch.from_numpy(label_test).type(torch.long)

        # Combine channels 
        XF_train = torch.stack([X_train,F_train], dim=1)
        XF_test = torch.stack([X_test,F_test], dim=1)

        X_train_list = [XF_train, X_train.unsqueeze(1)]
        X_test_list = [XF_test, X_test.unsqueeze(1)]

        # Train and test model with and without extra feature 
        for j in range(len(X_test_list)):

            train_dataset = torch.utils.data.TensorDataset(
                X_train_list[j],
                y_train
            )
            test_dataset = torch.utils.data.TensorDataset(
                X_test_list[j],
                y_test
                )
            
            # Create data loader and turn datasets into iterables (batches)
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

            # Initialize model and optimizer
            model =  model_builder.TVGG(
                input_shape=input_shapes[j], 
                hidden_units=10, 
                output_shape=2
                )
                        
            optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
            
            # Train test loop
            output = engine.train_test_loop(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer, 
                loss_fn=loss_fn,
                epochs=EPOCHS,
                print_b=False
                )
            
            acc_test = output['test_acc'][EPOCHS-1]
            ce_test = output['test_ce'][EPOCHS-1]            
            outputs.append((i,j, acc_test,ce_test))

    return (n, k, m,outputs)

################################################################################
#                             Perform simulations                              #
################################################################################

# Define variables for the simulation
# Number of replications and samples per class
NREPS = 100
n_samples = [[3000] * 2, [5000] * 2, [3000,5000]]

# Noise and error variance
noise_prop = [0,0.25,0.5,0.75,1]
VAR_ERROR =.15

# Parameters for extra feature
mu_c = np.array([-2,-1,-0.5,0])
mu_s = np.array([2,1,0.5,0])
sigma_s, sigma_c = 0.5, 0.5

# Define extra feature 
size_c = [
    np.zeros((len(mu_c), NREPS, n_samples[i][0]))
    for i in range(len(n_samples))
   ]

size_s = [
    np.zeros((len(mu_c), NREPS, n_samples[i][1]))
    for i in range(len(n_samples))
   ]

# Simulate extra feature
for n, n_sample in enumerate(n_samples):
    for k in range(NREPS):
        for i in range(len(mu_c)):
            size_c[n][i,k,] = np.random.normal(mu_c[i], sigma_c, n_sample[0])
            size_s[n][i,k,] = np.random.normal(mu_s[i], sigma_s, n_sample[1])

input_shapes = [2, 1]

# Simulate images 
data_noisy = [[[] for _ in range(len(n_samples))] for _ in range(len(noise_prop))]
for n, n_sample in enumerate(n_samples):
    for j, noise in enumerate(noise_prop):
        for k in range(NREPS):
            output = cs.generate_sample(
                n=n_sample, 
                noise_prop=noise,
                var=VAR_ERROR
            )
            data_noisy[j][n].append(output)
            
# Define Loss Function
loss_fn = nn.CrossEntropyLoss()

# Batch size and epochs
BATCH_SIZE = 50
EPOCHS = 10

# Initialize arrays to save errors
accuracy_test_noisy = np.zeros((len(n_samples),len(mu_c),len(noise_prop), 2 ,NREPS))
ce_test_noisy = np.zeros((len(n_samples),len(mu_c),len(noise_prop), 2 ,NREPS))

# Parallelize the loop
outputs_noisy = Parallel(n_jobs=100)(
    delayed(process_combination)(
        n, nsample, k, 
        data=data_noisy[m],
        size_c=size_c,
        size_s=size_s,
        nscenarios=len(mu_s),
        m=m
    )
    for n, nsample in enumerate(n_samples)
    for m, noise in enumerate(noise_prop)
    for k in range(NREPS)
)

for n, k, m, results in outputs_noisy:
    for i, j, acc, ce in results:
        accuracy_test_noisy[n,i,m,j,k]  = acc
        ce_test_noisy[n,i,m,j,k]  = ce        

# Save environment
env_vars = {
    'n_samples' : n_samples,
    'accuracy_test': accuracy_test_noisy,
    'ce_test': ce_test_noisy,
    'noise_prop': noise_prop,
    'NREPS': NREPS,
    'EPOCHS': EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'mu_c': mu_c,
    'mu_s': mu_s,
    'sigma_s': sigma_s,
    'sigma_c': sigma_c,
    'var_error': VAR_ERROR
}       
        
# Save variables using joblib 
joblib.dump(env_vars, '/u/ruizsuar/InformedMlCv/Environments/IF_Nov26.pkl')
                        
