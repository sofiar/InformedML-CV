################################################################################
################# Simulate data for two devices, train, ########################
############### test and save models for both data sets ########################
################################################################################

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modular import model_builder
import modular.samples_setup as cs
from modular import engine

# Simulate data
n_sample =50000

output_d1 = cs.generate_sample(n=n_sample, seed=11, noise_prop=1, var=0.35)
images_d1, labels_d1 = (output_d1['images'], output_d1['labels'])
train_dataset_d1, test_dataset_d1 = cs.generate_dataset(images_d1, labels_d1)

output_d2 = cs.generate_sample(n=n_sample, seed=11, noise_prop=1, var=0.05)
images_d2, labels_d2 = (output_d2['images'], output_d2['labels'])
train_dataset_d2, test_dataset_d2 = cs.generate_dataset(images_d2, labels_d2)

# Create data loaders
BATCH_SIZE = 32

train_dataloader_d1 = DataLoader(train_dataset_d1, 
batch_size=BATCH_SIZE, shuffle=True)

test_dataloader_d1 = DataLoader(test_dataset_d1,
batch_size=BATCH_SIZE, shuffle=False)

train_dataloader_d2 = DataLoader(train_dataset_d2, 
batch_size=BATCH_SIZE, shuffle=True)

test_dataloader_d2 = DataLoader(test_dataset_d2,
batch_size=BATCH_SIZE,shuffle=False)

# Train and test models for both devices
loss_fn = nn.CrossEntropyLoss()
epochs = 6
model_1 = model_builder.TVGG(input_shape = 1,
                            hidden_units = 10, 
                            output_shape = 2)                

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)
output_d1 = engine.train_test_loop(model_1,train_dataloader_d1,
                                   test_dataloader_d1, optimizer, loss_fn,
                                   epochs,print_b=False)

out_d1  = output_d1['test_acc'][epochs-1] 

model_2 = model_builder.TVGG(input_shape = 1,
                             hidden_units = 10, 
                             output_shape = 2)                

optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)
output_d2= engine.train_test_loop(model_2,train_dataloader_d2,
                                  test_dataloader_d2, optimizer, loss_fn,
                                  epochs,print_b=False)

out_d2 = output_d2['test_acc'][epochs-1]


print(f'Test error for device 1 is:', {out_d1})
print(f'Test error for device 2 is: ', {out_d2})

# Save models 
engine.save_model(model=model_2,
           target_dir='/home/sofia/Candu_postdoc/Informative_ML/models',
           model_name='model_d2.pth')
engine.save_model(model=model_1,
           target_dir='/home/sofia/Candu_postdoc/Informative_ML/models',
           model_name='model_d1.pth') 