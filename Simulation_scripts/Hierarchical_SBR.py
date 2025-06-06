"""Simulations to asses Hierarchical labels with SBR."""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import joblib
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split

from modular import engine
from modular import extra_functions as ef

# from modular import model_builder
# import modular.samples_setup as cs

################################################################################
############################### Sample setup ###################################
################################################################################

def generate_hierarchical_sample(n , noise_prop,seed = 999, size = 28, 
                                 var = 0.15, noise_indx = False):
        
    """Generates hierarchical samples.

     Args:
        n: vector of lenght amount of subclasses, containing the number of images of 
        each subclass to be generated in the following order: circles, ellipses, squares,
        rectangles and equilateral and isosceles.
        noise_prop: proportion of the data with noise
        seed: seed to perform the simulation. Default = 999.
        size: size of the image as number of pixels.
        var: variance for the Gaussian Error. Default = 0.15.
        noise_indx: Boolean indicating wheather the indexes of noisy samples 
        should be return. Default= False.
       
    Returns:
        A Dictionary of Arrays of images and labels simulated.
        In the form (images, sub_labels,main_lables)
        and (images, sub_labels,main_lables, noisy index) if noise_indx =True
    
  """
  
    dataset_circles = []
    dataset_squares = []
    dataset_triangles = []
        
    ntot = sum(n)
    n_cr = n[0]
    n_sq = n[1]
    n_tr = n[2]
                    
    # Create circles
    for _ in range(n_cr):
        radius = np.random.randint(2, size // 2)  #
        image = ef.draw_circle(size, radius)
        dataset_circles.append(image)
            
    # Create squares
    for _ in range(n_sq):
        len_size = np.random.randint(2, size )  
        image = ef.draw_square(size, len_size)
        dataset_squares.append(image)
        
    # Create triangles
    for _ in range(n_tr):
        len_size = np.random.randint(2, size)  
        image = ef.draw_triangle(size, len_size)
        dataset_triangles.append(image)
     
    ## Add noise to some of them
    np.random.seed(seed)
     
    # noisy circles      
    noisy_indices_c = np.random.choice(n_cr, int(n_cr * noise_prop), replace=False)
    for i in noisy_indices_c: 
        dataset_circles[i] = ef.add_gaussian_noise(dataset_circles[i],var=var)
       
    # noisy squares
    noisy_indices_s = np.random.choice(n_sq, int(n_sq * noise_prop), replace=False)
    for i in noisy_indices_s:
        dataset_squares[i] = ef.add_gaussian_noise(dataset_squares[i],var=var)
        
    # noisy triangles
    noisy_indices_is = np.random.choice(n_tr, int(n_tr * noise_prop), replace=False)
    for i in noisy_indices_is:
        dataset_triangles[i] = ef.add_gaussian_noise(dataset_triangles[i],var=var)    
        
    # Create sub labels
    circle_labels = np.full(n_cr, 0)
    square_labels = np.full(n_sq, 1)
    traingle_labels = np.full(n_tr, 2)
        
    curved_labels = np.full(n_cr , 0)
    polygon_labels = np.full(n_sq + n_tr, 1)
    
    # # Combine datasets
    all_images = np.concatenate((dataset_circles , dataset_squares,
                                 dataset_triangles))
                                    
    all_sublabels = np.concatenate((circle_labels, square_labels, 
                                    traingle_labels))

    all_mainlabels = np.concatenate((curved_labels,polygon_labels))
           
    # Shuffle the dataset
    indices = np.arange(len(all_images))
    np.random.seed(seed)
    np.random.shuffle(indices)
    all_images = all_images[indices]
    all_sublabels = all_sublabels[indices]
    all_mainlabels = all_mainlabels[indices]
    noisy_indx  = np.concatenate((noisy_indices_c,ntot + noisy_indices_s))
    noisy_indx = np.nonzero(np.isin(indices, noisy_indx))[0]
        
    # recover noisy indices
    if noise_indx:
        noisy_indx  = np.concatenate((noisy_indices_c,ntot + noisy_indices_s))    
        output = {"images": all_images,
                  "sublabels": all_sublabels,
                  "mainlabels": all_mainlabels,
                  "indx_noisy_data": noisy_indx}
    else:
        output = {"images": all_images,
                "sublabels": all_sublabels,
                "mainlabels": all_mainlabels}
    return output

def generate_hierarchical_dataset(all_images, all_sublabels, all_mainlabels, 
                                  test_size=0.2,seed = 999):

    """Generate Hierarchical dataset from data sample.

     Args:
        all_images: A array with images generated from `generate_hierarchical_sample`
        all_sublabels: A array of sub labels corresponding to the images,
        generated from`generate_hierarchical_sample`
        all_mainlabels: A array of main labels corresponding to the images,
        generated from`generate_hierarchical_sample`
        test_size: proportion of of the total amount of images to be used for 
        testing. Default = 0.2.
        seed: seed to perform the simulation. Default = 999.

        
    Returns:
        A tuple of TensorDatasets of train and test data sets.
        In the form (train_dataset, test_dataset)
    
  """
    
    X = torch.from_numpy(all_images).type(torch.float)
    Y_main = torch.from_numpy(all_mainlabels).type(torch.long)
    Y_sub = torch.from_numpy(all_sublabels).type(torch.long)

    (
     X_train, X_test, y_main_train, 
     y_main_test, y_sub_train, y_sub_test
    ) = train_test_split(
        X, Y_main,Y_sub, test_size = test_size, random_state = seed
    ) 

    # Add channel at dimension 1 (greyscale)
    X_train = X_train.unsqueeze(1)  
    X_test = X_test.unsqueeze(1)
        
    train_dataset_ = torch.utils.data.TensorDataset(
        X_train,y_main_train, y_sub_train
    )
    test_dataset_ = torch.utils.data.TensorDataset(
        X_test,y_main_test,y_sub_test
    )

    return train_dataset_, test_dataset_

################################################################################
############################### semantic loss  #################################
################################################################################

def semantic_regularization_loss(logits_main,true_sublabels,coef_lambda):
    """
    Implements the semantic rule:
    POLYGON(x) ⇒ SQUARE(x) ∧ TRIANGLE(x)
    CURVED(x) ⇒ CIRCLE(x)
        
     Args:
        logits_main (Tensor): vector containing the model's logits for the 
        main classes
        true_sublabels (Tensor): vector containing the tru sublabels 
       
    Returns: 
            float: the value of the regularized loss function 
    """
   
    if coef_lambda is None:
        coef_lambda = [1,1,1]
        
    # Apply softmax to logits to get probabilities
    probs_main = F.softmax(logits_main, dim=1)
    
    # Probabilities for each main class
    p_curved = probs_main[:,0]
    p_polygon = probs_main[:,1]
    
    # violations
    violation_1 = torch.mean((true_sublabels == 0) * p_polygon )
    violation_2 = torch.mean((true_sublabels == 1) * p_curved)
    violation_3 = torch.mean((true_sublabels == 2) * p_curved)
               
    reg_term = (coef_lambda[0] *violation_1.mean() + 
                coef_lambda[1] * violation_2.mean() + 
                coef_lambda[2] * violation_3.mean()) 
    
    
    return reg_term


def train_step_reg(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fun: torch.nn.Module,
               reg_fn,
               optimizer: torch.optim.Optimizer,
               alpha: float,
               device: torch.device = None,
               coef_lambda = None)-> Tuple[float, float]:
    
    """Trains a PyTorch model with hierarchical labels including 
    a regularization term in the loss fuction for 1 epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps 

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fun: A PyTorch base loss function to minimize.
        reg_fn: regularized function with inputs, logits_main and logits_sub 
        optimizer: A PyTorch optimizer to help minimize the loss function.
        alpha: weight for the regularization term
        device: Torch device

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy)
    
  """
    model.to(device)
    model.train()
    train_loss, train_acc, train_ce= 0, 0, 0

    for batch, (X, y_main, y_sub) in enumerate(dataloader):
        X, y_main, y_sub = X.to(device), y_main.to(device), y_sub.to(device)
        y_predmain, y_predsub = model(X)

        # Optimizer zero grad
        optimizer.zero_grad()
        
        # Calculate standard loss
        base_loss = loss_fn(y_predsub, y_sub) +  loss_fn(y_predmain, y_main)
        
        # add regularization term
        sbr_loss = reg_fn(
            logits_main = y_predmain,
            true_sublabels = y_sub,
            coef_lambda = coef_lambda
        )
        
        loss = base_loss + alpha * sbr_loss
        
        train_loss += loss    
        
        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

       # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_predsub, dim=1), dim=1)
        train_acc += (y_pred_class == y_sub).sum().item()/len(y_pred_class)

        # Calculate Cross entropy
        train_ce += engine.cross_entropy_fn(y_true=y_sub, y_preds=y_predsub)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_ce = train_ce/ len(dataloader)
    return train_loss, train_acc, train_ce


def test_step_reg(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fun: torch.nn.Module,
               reg_fn,
               alpha: float,
               device: torch.device = None,
               coef_lambda = None)-> Tuple[float, float]:
   
    """Test a PyTorch model for 1 epoch.

    Turns a target PyTorch model to eval mode and then
    runs forward pass on the test set

    Args:
        model: A PyTorch model to be used
        dataloader: A DataLoader instance for testing the model
        loss_fun: A PyTorch loss function to minimize.
        reg_fn: regularized function with inputs, logits_main and logits_sub 
        alpha: weight for the regularization term.
        device: Torch device


    Returns:
        A tuple of test loss and test accuracy metrics.
        In the form (test_loss, test_accuracy)
    
  """
    test_loss, test_acc, test_ce = 0, 0, 0
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for X, y_main, y_sub in dataloader:
            X, y_main, y_sub = X.to(device), y_main.to(device), y_sub.to(device)

            # Forward pass
            test_predmain, test_predsub = model(X)
                      
            sbr_loss = reg_fn(
                logits_main = test_predmain,
                true_sublabels = y_sub,
                coef_lambda = coef_lambda
            )
           
            base_loss = (
                loss_fn(test_predsub, y_sub) +  loss_fn(test_predmain, y_main)
            )
            loss = base_loss + alpha * sbr_loss
                              
            test_loss += loss  
           
            # Calculate accuracy over subclasses
            test_acc += engine.accuracy_fn(
                y_true=y_sub, 
                y_pred=test_predsub.argmax(dim=1)
            )

            # Calculate Cross entropy
            test_ce += engine.cross_entropy_fn(
                y_true = y_sub,
                y_preds=test_predsub
            ) 

        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        test_ce /= len(dataloader)
        
        
    return test_loss, test_acc, test_ce


def train_test_loop_reg(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fun: torch.nn.Module,
          reg_fn,
          epochs: int,
          print_b: True ,
          alpha: float,
          device: torch.device = None,
          scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          coef_lambda = None) -> Dict[str, List]:   
    
    """ Train test loop with regularization term by epochs.

    Conduct train test loop 

    Args:
        model: A PyTorch model to be used
        train_dataloader: A DataLoader instance for trainig the model
        test_dataloader: A DataLoader instance for testinig the model
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch base loss function to minimize.
        reg_fn: regularized function with inputs, logits_main and logits_sub 
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
        train_loss, train_acc, train_ce = train_step_reg(model = model,
                                           dataloader = train_dataloader,
                                           loss_fun = loss_fun,
                                           reg_fn = reg_fn,
                                           optimizer = optimizer,
                                           alpha = alpha,
                                           device = device,
                                           coef_lambda = coef_lambda)
        
        test_loss, test_acc, test_ce = test_step_reg(model = model, 
                                           dataloader = test_dataloader,
                                           loss_fun = loss_fun,
                                           reg_fn = reg_fn,
                                           alpha = alpha,
                                           device = device,
                                           coef_lambda = coef_lambda)
        
      # Adjust learning rate
        if scheduler is not None:
            scheduler.step()  

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

################################################################################
############################### Adapeted TinyVgg  ##############################
################################################################################

# Define TinyVGG model for hierarchical labels
class HierarchicalTVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    
    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, 
                 num_main_clases: int,
                 num_sub_clases: int,
                 resolution: int = 28):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1,      
                      padding=1),    
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)   
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Flatten layer before classification
        self.flattern = nn.Flatten()
        flattern_size = hidden_units * (resolution//4) * (resolution//4)
        
        # Output layers: main class and subclass
        self.fc_main = nn.Linear(flattern_size,num_main_clases)
        self.fc_sub = nn.Linear(flattern_size,num_sub_clases)
        
        
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.flattern(x)
        
        # Separate outputs for main category and subcategory
        y_main = self.fc_main(x)
        y_sub = self.fc_sub(x)

        return y_main, y_sub
    
    
def get_predictions_hierarchical(model, image, true_label ,probs= False):
    
    """Extracts predicted labels based on hierarchical structure."""

    with torch.no_grad():  
        out_main, out_sub = model(image)  # Forward pass
        if probs:
            probabilities = F.softmax(out_sub, dim=1)  # Convert logits to probabilities
        pred_label = out_sub.argmax(dim=1)  # Get predicted class indices (tensor)
        
    if probs:
        return true_label, pred_label, probabilities 
    else:
        return true_label, pred_label 
    
################################################################################
################################# Simulation  ##################################
################################################################################
print('Running')

SEED = 1222
BATCH_SIZE = 50
RESOLUTION = 50
EPOCHS = 40
NREPS = 30 
NOISE_PROP = 0.8
VAR = 0.5
NJOBS = 1
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# To reduce variability when re-running 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. Generate sample
n_samples = [3000,2000,1000]
#n_samples = [2500,2500,2500]
alphas = [0,0.001,0.003,0.005,0.008,0.01]


# 4. Start empty lists to save results 
trues_list = []
preds_list =[]
probs_list = []
alphas_list = []
nrep_list = []

for j in range(NREPS):
    for a_indx, a_val in enumerate(alphas):
        
        # 1. Simulate data
        simulated_sample = generate_hierarchical_sample(
                            n = n_samples, noise_prop = NOISE_PROP, 
                            size = RESOLUTION, var = VAR,
                            noise_indx = False)

        images, sublabels, mainlabels= (simulated_sample['images'], 
                                        simulated_sample['sublabels'],
                                        simulated_sample['mainlabels'])
        
        # 2. Generate dateset
        train_dataset, test_dataset = generate_hierarchical_dataset(
                                        all_images = images, 
                                        all_mainlabels = mainlabels,
                                        all_sublabels = sublabels,
                                        test_size = 0.20)

        # 3. Generate dataloader
        train_dataloader = DataLoader(train_dataset, batch_size= BATCH_SIZE, 
                            shuffle = False)
        
        test_dataloader = DataLoader(test_dataset, batch_size= BATCH_SIZE,
                            shuffle = False)
        
        # 4. Initialize model, optimer and base loss funtion
        model_tvgg = HierarchicalTVGG(input_shape = 1,
                                hidden_units = 10, 
                                num_main_clases = 2,
                                num_sub_clases = 3,
                                resolution = RESOLUTION)

        optm = torch.optim.SGD(params=model_tvgg.parameters(), lr=0.05)
        schedr = torch.optim.lr_scheduler.StepLR(optm, step_size=5, 
                                                    gamma=0.1)
        loss_fn = nn.CrossEntropyLoss()
        
        # 5. Train model
        output = train_test_loop_reg(model = model_tvgg,
                                    train_dataloader = train_dataloader,
                                    test_dataloader = test_dataloader,
                                    optimizer = optm,
                                    loss_fun = loss_fn, 
                                    reg_fn = semantic_regularization_loss,
                                    epochs = EPOCHS, 
                                    device = my_device,
                                    print_b = False,
                                    alpha = a_val,
                                    scheduler = schedr,
                                    coef_lambda=[1,1,1])
                                    #coef_lambda=[0.5,1,1])
        
        # 6. Simulate new test set with no missing sub labels 
        new_simulated_sample = generate_hierarchical_sample(
                                n = [500]*3, noise_prop = 0.5, 
                                size = RESOLUTION, var = VAR,
                                noise_indx = False)
        
        test_images, test_sublabels, test_mainlabels = (
                                        new_simulated_sample['images'], 
                                        new_simulated_sample['sublabels'],
                                        new_simulated_sample['mainlabels']
                                        )
        
        # 6.2 Get DataLoader
        X = torch.from_numpy(test_images).type(torch.float)
        X = X.unsqueeze(1)  
        Y_main = torch.from_numpy(test_mainlabels).type(torch.long)
        Y_sub = torch.from_numpy(test_sublabels).type(torch.long)

        new_test_dataset = torch.utils.data.TensorDataset(X,Y_main,Y_sub)
       
        new_test_dataloader = DataLoader(new_test_dataset,
                                        batch_size= BATCH_SIZE,
                                        shuffle = False)
        
        # 7. Get predictions 
        preds = (Parallel(n_jobs=NJOBS)(delayed(get_predictions_hierarchical)(
                                            model= model_tvgg,
                                            image=imag.to(my_device),
                                            true_label=targetsub,
                                            probs=True)
                        for imag, targetmain, targetsub  in new_test_dataloader))
    
     
        for idx, prds in enumerate(preds): 
        #for i in range(len(preds)):
            label, y_pred, probas =  prds
            trues_list.append(label)
            preds_list.append(y_pred)
            probs_list.append(probas)
            alphas_list.append([a_val]*len(label))
            nrep_list.append([j]*len(label))

################################################################################
################################ Save results ##################################
################################################################################    


# Save environment
env_vars = {'n_samples' : n_samples,
            'alphas': alphas_list,
            'true_labels': trues_list,
            'pred_probs': probs_list,
            'predicted_labels': preds_list,
            'nrep_list': nrep_list,
            'NREPS': NREPS,
            'EPOCHS': EPOCHS,
            'BATCH_SIZE': BATCH_SIZE,
            'NOISE_PROP': NOISE_PROP,
            'VAR_NOISE': VAR
            }

# Save variables using joblib 
# where_to_save = ('/home/sofiruiz/InformedMlCv/Environments/'
#                  'SBR_Hierarchical_cc_balanced.pkl')
where_to_save = ('/home/sofiruiz/InformedMlCv/Environments/'
                 'SBR_Hierarchical_cc.pkl')
torch.save(env_vars, where_to_save)

print('done!')