### Semantic Based Regularization ###
import sys
import os
from typing import Dict, List, Tuple
import joblib
from joblib import Parallel, delayed
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

modular_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(modular_path)
from modular import engine
import modular.samples_setup as cs
from modular import model_builder

# set number of cores
torch.set_num_threads(100)

# Define functions 
def semantic_regularization_loss(logits):
    """
    Implements the semantic rule:
    CIRCLE(x) ⇒ ¬TRIANGLE(x) ∧ ¬SQUARE(x)
    TRIANGLE(x) ⇒ ¬CIRCLE(x) ∧ ¬SQUARE(x)
    SQUARE(x) ⇒ ¬TRIANGLE(x) ∧ ¬CIRCLE(x)
    """
    # Apply softmax to logits to get probabilities
    probs = F.softmax(logits, dim=1)

    # Probabilities for each class
    p_circle, p_square, p_triangle = probs[:, 0], probs[:, 1], probs[:, 2]

    # Rule: 
    violation_1 = p_circle * (p_triangle + p_square)
    violation_2 = p_triangle * (p_circle + p_square)
    violation_3 = p_square * (p_triangle + p_circle)

    reg_term = violation_1.mean() + violation_2.mean() + violation_3.mean()
    return reg_term

def train_step_reg(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               alpha: float)-> Tuple[float, float]:
    """Trains a PyTorch model for 1 epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps 

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy)
    
  """
    model.train()
    train_loss, train_acc, train_ce= 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        y_pred = model(X)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Calculate standard loss
        base_loss = loss_fn(y_pred, y)
        # add regularization term
        sbr_loss = semantic_regularization_loss(y_pred)
        loss = base_loss + alpha * sbr_loss
        
        train_loss += loss    # Accumulatively add up the loss per epoch
        
        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

       # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        # Calculate Cross entropy
        train_ce += engine.cross_entropy_fn(y_true=y.detach().numpy(),
                                     y_preds=y_pred.detach().numpy()).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_ce = train_ce/ len(dataloader)
    return train_loss, train_acc, train_ce


def train_test_loop_reg(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          print_b: True ,
          alpha= float) -> Dict[str, List]:   
    """ Train test loop by epochs.

    Conduct train test loop 

    Args:
        model: A PyTorch model to be used
        train_dataloader: A DataLoader instance for trainig the model
        test_dataloader: A DataLoader instance for testinig the model
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to minimize.
        epochs: Number of epochs to run
        print_b: Boolean. When True the epochs and the test accuracy is printed. 


    Returns:
        A list of train loss, train accuracy metrics, test loss,
        test accuracy metrics.
        In the form (train_loss, train_accuracy,test_loss, test_accuracy)
    
  """
    results = {"train_loss": [],
               "train_acc": [],
               "train_ce": [],
               "test_loss": [],
               "test_acc": [],
               "test_ce": []}
                    
    for epoch in range(epochs):
        train_loss, train_acc, train_ce = train_step_reg(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           alpha = alpha)
        
        test_loss, test_acc, test_ce = engine.test_step(model=model, dataloader=test_dataloader,
                                        loss_fn=loss_fn)

      # Print out what's happening
        if print_b:
            print(
                f"Epoch: {epoch+1} | "
                f"test_acc: {test_acc:.4f}"
            )

      # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_ce"].append(train_ce)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_ce"].append(test_ce)

    return results

# Parallelize loop 
def process_combination(v, var, n, nsample, k, p_errors, alphas):
    noise_p = p_errors[v]
    # 1. Simulate images and labels
    output = cs.generate_sample(n = nsample,  
                                noise_prop = noise_p, var=var,
                                Nclass=3)
    images, labels= (output['images'], output['labels'])
    
    # 2. Generate dataset
    train_dataset, test_dataset = cs.generate_dataset(images, labels)
    train_dataloader = DataLoader(train_dataset, 
                        batch_size= 32, 
                        shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                        batch_size= 32,
                        shuffle=True )
                
    # 3. Load model
    model = model_builder.TVGG(input_shape = 1,
                        hidden_units = 10, 
                        output_shape = 3)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    outputs = []
    for a, alpha in enumerate(alphas):
        # 4. Run model
        output = train_test_loop_reg(model = model,
                                    train_dataloader = train_dataloader,
                                    test_dataloader = test_dataloader,
                                    optimizer = optimizer,
                                    loss_fn = loss_fn, 
                                    epochs = EPOCHS, 
                                    print_b=False,
                                    alpha = alpha)

        acc_test = output['test_acc'][EPOCHS-1]
        ce_test = output['test_ce'][EPOCHS-1]            
        outputs.append((a, acc_test,ce_test))

    # Print message 
    if k % 50 == 0:  
        print(f"Combination ({var}, {nsample}) done!")
   
    return (v, n, k, outputs)

            
    
# Define variables

alphas = [0,0.1,0.5,1,1.5,3]#[0,0.05,0.1,0.2,0.3]
NREPS = 50
n_samples = [[800,3000,3200],[3000,1000,5000],[2500]*3,[5000]*3]
var_errors = [0,0.15,0.15] 
prop_errors = [0,0.5,1]

# Keep ce and test accuracy
accuracy_test = np.zeros((len(var_errors),len(alphas), len(n_samples) ,NREPS))
ce_test = np.zeros((len(var_errors),len(alphas), len(n_samples) ,NREPS))

BATCH_SIZE = 50
EPOCHS = 10


# Parallelize the outermost loop
outputs = Parallel(n_jobs=100)(delayed(process_combination)(v, var, n, nsample,  
                               k, p_errors = prop_errors,
                               alphas = alphas)
                    for v, var in enumerate(var_errors)
                    for n, nsample in enumerate(n_samples)
                    for k in range(NREPS))

for v, n, k, results in outputs:
    for a, acc, ce in results:
        accuracy_test[v,a,n,k]  = acc
        ce_test[v,a,n,k]  = ce


# for v, n, a, k, ac_t, ce_t in outputs:
#     accuracy_test[v,a,n,k]  = ac_t
#     ce_test[v,a,n,k]  = ce_t
     


# Save environment
env_vars = {'n_samples' : n_samples,
            'var_errors': var_errors,
            'alphas': alphas,
            'accuracy_test': accuracy_test,
            'ce_test': ce_test,
            'NREPS': NREPS,
            'EPOCHS': EPOCHS,
            'BATCH_SIZE': BATCH_SIZE
            }

# Save variables using joblib 
joblib.dump(env_vars, '/u/ruizsuar/InformedML-CV/Environments/SBR_Nov25.pkl')

