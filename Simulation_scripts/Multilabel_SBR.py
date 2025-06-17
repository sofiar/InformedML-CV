import torch 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

from torch.utils.data import DataLoader, random_split

from modular import engine
from modular import extra_functions as ef
from modular import model_builder
import modular.samples_setup as cs
from torch.utils.data import Dataset, Subset

from torchmetrics.classification import (
    MultilabelAccuracy, 
    MultilabelPrecision, 
    MultilabelRecall,
    MultilabelF1Score
)

################################################################################
##################### classes and functions definitions ########################
################################################################################

def generate_multilabel_sample(n , noise_prop,seed = 999,size = 28, var = 0.15,
                    noise_indx = False):
    
    """Generates multi-label samples.

     Args:
        n: vector, number of elements of each class
        noise_prop: proportion of the data with noise
        seed: seed to perform the simulation. Default = 999.
        size: size of the image as number of pixels.
        var: variance for the Gaussian Error. Default = 0.15.
               
    Returns:
        A Dictionary of Arrays of images and labels simulated.
        In the form (images, lables)
    
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
        len_size = np.random.randint(2, size)  
        image = ef.draw_square(size, len_size)
        dataset_squares.append(image)
        
    # Create triangles
    for _ in range(n_tr):
        len_size = np.random.randint(2, size )  
        image = ef.draw_triangle(size, len_size)
        dataset_triangles.append(image)
                        
    ## Add noise to some of them
    np.random.seed(seed)
     
    # circles      
    noisy_indices_c = np.random.choice(n_cr, int(n_cr * noise_prop), replace=False)
    for i in noisy_indices_c: 
        dataset_circles[i] = ef.add_gaussian_noise(dataset_circles[i],var=var)
        
    # squares
    noisy_indices_s = np.random.choice(n_sq, int(n_sq * noise_prop), replace=False)
    for i in noisy_indices_s:
        dataset_squares[i] = ef.add_gaussian_noise(dataset_squares[i],var=var)
        
    # triangles
    noisy_indices_is = np.random.choice(n_tr, int(n_tr * noise_prop), replace=False)
    for i in noisy_indices_is:
        dataset_triangles[i] = ef.add_gaussian_noise(dataset_triangles[i],var=var)
        
    # Crete dictionary 
    class_to_indx = dict(zip(['circles','squares','triangles','curved','Polygons'],
                             [0,1,2,3,4,5]))
    
    # Create labels
    size = (ntot,5)
    labels = np.zeros(size)
    labels[0:n_cr,(0,3)] = 1
    labels[n_cr:(n_cr+n_sq),(1,4)] = 1
    labels[(n_cr+n_sq):(n_cr+n_sq)+n_tr,(2,4)] = 1
    
    # # Combine datasets
    all_images = np.concatenate((dataset_circles , dataset_squares,
                                 dataset_triangles))
    # Covert to three channels
    all_images_3c = []
    for i in range(len(all_images)):
        all_images_3c.append(np.stack([all_images[i]] * 3, axis=0))
   
    # Shuffle the dataset
    indices = np.arange(len(all_images))
    np.random.seed(seed)
    np.random.shuffle(indices)
    all_images_3c = [all_images_3c[i] for i in indices]
    labels = labels[indices]
    output = {
        "images": all_images_3c,
        "labels": labels,
        "class_to_indx": class_to_indx
            }    
    
    return output

class MultilabelDataSet(Dataset):
    def __init__(self,images,labels,transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, idx):
        # Load image
        image = self.images[idx]
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            
        #Get label
        label = self.labels[idx]    
        
        return image, label
    
def reg_loss(y_hat, coef_lambda=None):
    """
    Implements the semantic rule:
    TRIANGLE(x) ⇒ ¬ CURVED(x)
    SQUARE(x) ⇒ ¬ CURVED(x)
    CIRCLE(x) ⇒ ¬ POLYGON(x)

    Args:
        y_hat (Tensor): vector containing the model's output
        coef_lambda (list): OPTIONAL, coefficients for each violation
        
    Returns: 
            float: the value of the regularized loss function 
    """

    if coef_lambda is None:
        coef_lambda = [1,1,1]
        
    out_sigmoid = torch.sigmoid(y_hat)
    circles = out_sigmoid[:,0]
    squares = out_sigmoid[:,1]
    triangles = out_sigmoid[:,2]
    curved = out_sigmoid[:,3]
    polygons = out_sigmoid[:,4]


    violation1 = torch.mean(circles * polygons)
    violation2 = torch.mean(squares * curved)
    violation3 = torch.mean(triangles * curved)

    reg_term = (coef_lambda[0]*violation1 + 
                coef_lambda[1]*violation2 +
                coef_lambda[2]*violation3 )  

    return reg_term


def train_val_loop(EPOCHS,model,train_dataloader,
                   val_dataloader, device,
                   optimizer,reg_fn,alpha,coef_lambda = [1,1,1],
                   verbose = False):
    
    """Trains and validates a PyTorch model including regularization function.

    Args:
        EPOCHS: int. number of epochs to run
        model: A PyTorch model to be trained.
        train_dataloader: A DataLoader instance for the model to be trained on.
        val_dataloader: A DataLoader instance for the model to be validated on.
        device: torch.device
        optimizer: A PyTorch optimizer to help minimize the loss function.
        reg_fn: callable.  The regularization function applied to the model
        alpha: float. Weightening factor for the regularization term
        coef_lambda: list of floats, default = [1,1,1]. Coefficients controlling 
        the weight for each constrain violation. 
        verbose: bool. If true prints the training and validation loss per epoch
         
    Returns:
        training loss and validation loss.
        In the form (training_loss, validation_loss)
    
  """
    model.to(device)
    epoch_train_loss = []
    epoch_val_loss = []
    
    # Train Model
    for epoch in range(EPOCHS):
        train_losses = []
        model.train()
        for batch,(X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            X = X.to(device,dtype = torch.float)
            y = y.to(device,dtype = torch.float)
            y_hat = model(X)
            error = nn.BCEWithLogitsLoss()
            loss = torch.sum(error(y_hat,y)) + alpha * reg_fn(y_hat,coef_lambda)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        epoch_train_loss.append(np.mean(train_losses))
        
        # Validation step
        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch,(X, y) in enumerate(val_dataloader):
                X = X.to(device,dtype = torch.float)
                y = y.to(device,dtype = torch.float)
                y_hat = model(X)
                error = nn.BCEWithLogitsLoss()
                loss = torch.sum(error(y_hat,y)) + alpha * reg_loss(y_hat,coef_lambda)
                val_losses.append(loss.item())
        epoch_val_loss.append(np.mean(val_losses))
        
        if verbose:
            if epoch%10==0:

                print(
                    'Train Epoch: {}\t Train Loss: {:.6f}\t Val Loss: {:.6f}'.format(
                        epoch+1,
                        np.mean(train_losses),
                        np.mean(val_losses))               
                    )      
            
    return(epoch_train_loss,epoch_val_loss)      

def get_predictions(model,image):
         
    """Calculates predictions from a trained multi-label PyTorch model. It takes 
    the two categories with largest confidence.  
    
    Args:
        model: A PyTorch trained model 
        image: new sample to be predicted  
    
    Returns:
        predictions and confidence 
    
  """
    out = model(image)
    probs = torch.sigmoid(out)
    # to get predictions by threshold
    # preds = (probs>thr).float()
    # to get predictions by top 2
    preds = torch.zeros_like(probs)  
    topk = torch.topk(probs, k=2, dim=1)  
    topk_indices = topk.indices   
    preds.scatter_(1, topk_indices, 1.0)    
     
    return preds, probs


def test_loop(model,dataloader):
    
    """Generate predictions for samples in the test Dataloader from a trained 
    multi-label model   
   
    Args:
        model: A PyTorch trained model 
        dataloader: Test dataloader to be predicted  
    
    Returns:
        lists of real labels, predicted labels and confidence 
    
    """
    model.to(device)
    model.eval()
   
    real_labels = []
    pred_labels = []
    probs_labels = []
    with torch.no_grad():
        for batch,(X, y) in enumerate(dataloader):
                X = X.to(device,dtype = torch.float)
                y = y.to(device,dtype = torch.float)
                preds, probs = get_predictions(model,X)
                pred_labels.append(preds)
                probs_labels.append(probs)
                real_labels.append(y)
            
    return real_labels, pred_labels, probs_labels    

################################################################################
########################### perform simulatinons ###############################
################################################################################

print('Running')

SEED = 1222
BATCH_SIZE = 50
EPOCHS = 40
NREPS = 30 
NOISE_PROP = 0.8
VAR = 0.2
NJOBS = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# To reduce variability when re-running 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Generate sample
nsamples = [[1000] * 3,
                 [500, 1000, 1500],
                 [2500] * 3,
                 [500,1500,2000]]
alphas = [0,0.05,0.07,0.1,0.15]

# Start empty lists to save results 
accuracy_list = []
recall_list =[]
precision_list = []
f1_list = []
alphas_list = []
nrep_list = []
pol_circ = []
cur_sqr = []
cur_tri = []
n_samples_list = []

# Metrics
accuray_metric = MultilabelAccuracy(num_labels=5).to(device)
precision_score = MultilabelPrecision(num_labels=5).to(device)
recall_score = MultilabelRecall(num_labels=5).to(device)
f1_score = MultilabelF1Score(num_labels=5).to(device)

for ns_indx, n_sample in enumerate(nsamples):
    for j in range(NREPS):
        for a_indx, a_val in enumerate(alphas):
            
            # 1. Simulate data
            simulated_sample = generate_multilabel_sample(
                n = n_sample, 
                noise_prop = NOISE_PROP, 
                var = VAR
            )

            # 2. Generate dateset
            data_class = MultilabelDataSet(
                simulated_sample['images'],
                simulated_sample['labels']
            )

            # 3. Split train, test, validation
            train_size = int(0.7 * len(data_class.labels))
            val_size = int(0.10 * len(data_class.labels))
            test_size = len(data_class.labels) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                data_class, 
                [train_size, val_size, test_size]
            )
            
            # 4. Generate dataloaders
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size = BATCH_SIZE,
                shuffle = True
            )

            val_dataloader = DataLoader(
                val_dataset, 
                batch_size = BATCH_SIZE,
                shuffle = False
            )

            test_dataloader = DataLoader(
                test_dataset, 
                batch_size = BATCH_SIZE,
                shuffle = False
            )
            
            # 5. Initialize model and optimer 
            model = model_builder.TVGG(
                    input_shape = 3,  
                    hidden_units= 10, 
                    output_shape = 5
                )   
            optimizer = torch.optim.Adam(params = model.parameters(), lr=0.001)
            
            # 6. Train model
            train_loss, val_loss = train_val_loop(
                EPOCHS = EPOCHS, 
                model = model, 
                train_dataloader = train_dataloader,
                val_dataloader = val_dataloader, 
                device = device,
                optimizer = optimizer,
                reg_fn = reg_loss,
                alpha = a_val,
                coef_lambda = [1,.85,.2]
            )
            
            # 7. Inference
            real_labels, pred_labels, probs_labels = test_loop(
                model = model,
                dataloader = test_dataloader
            )
                            
            all_probs = torch.cat(probs_labels, dim=0)  
            all_true_labels = torch.cat(real_labels, dim=0)
            all_pred_labels = torch.cat(pred_labels, dim=0)

            # Save metrics     
            accuracy_list.append(accuray_metric(all_probs, all_true_labels))
            recall_list.append(recall_score(all_probs, all_true_labels))
            precision_list.append(precision_score(all_probs, all_true_labels))
            f1_list.append(f1_score(all_probs, all_true_labels))
            alphas_list.append(a_val)
            nrep_list.append(a_indx)
            n_samples_list.append(ns_indx)
            
            # Save number of contradictions  
            pred_numpy = torch.cat(pred_labels, dim=0).cpu().numpy()
            real_numpy = torch.cat(real_labels,dim=0).cpu().numpy()
            pol_circ.append(
                np.sum(np.all(pred_numpy == np.array([1,0,0,0,1]), axis=1))
            )
            cur_sqr.append(
                np.sum(np.all(pred_numpy == np.array([0,1,0,1,0]), axis=1))
            )
            cur_tri.append(
                np.sum(np.all(pred_numpy == np.array([0,0,1,1,0]), axis=1))
            )

################################################################################
################################ Save results ##################################
################################################################################    

# Save environment

env_vars = {'nsamples' : nsamples,
            'alphas': alphas_list,
            'accuracy': accuracy_list,
            'recall': recall_list,
            'precision': precision_list,
            'f1': f1_list,
            'nrep_list': nrep_list,
            'nsamples_list': n_samples_list,
            'pol_circ': pol_circ,
            'cur_sqr': cur_sqr,
            'cur_tri': cur_tri,             
            'EPOCHS': EPOCHS,
            'BATCH_SIZE': BATCH_SIZE,
            'NOISE_PROP': NOISE_PROP,
            'VAR_NOISE': VAR
            }

# Save variables using joblib 
where_to_save = ('/home/sofiruiz/InformedMlCv/Environments/'
                 'SBR_Multilabel.pkl')
torch.save(env_vars, where_to_save)

print('done!')


    
    
