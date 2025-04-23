
"""
Contains functions and classes for training, testing and saving a PyTorch model.
"""
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        
    """ Args: 
        delta(float) : Minimum change in the monitored quantity to qualify as an improvement.
        patience(int): Number of epochs to wait before stopping if no improvement.
        
     """    

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
            
     

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """ Defines accuracy measure as the percentage of samples well classified"""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def cross_entropy_fn(y_true,y_preds):
    """ Defines cross entropy measure """
    
    device = y_true.device  # Get the device of y_true (CPU or GPU)
       
    # One-hot encode y_true directly using PyTorch without needing np.eye
    y_true = torch.nn.functional.one_hot(y_true, num_classes=y_preds.shape[1]).float().to(device)
    
    # Clip predictions to avoid log(0)
    y_preds = torch.clamp(y_preds, min=torch.finfo(torch.float32).eps, max=1 - torch.finfo(torch.float32).eps)

    # Compute cross-entropy for each observation
    ce = -torch.sum(y_true * torch.log(y_preds), dim=1)
    
    # Return average cross-entropy loss as a scalar (move back to CPU if using GPU)
    return ce.mean().item()

    # Numpy cpu version
    # y_true = np.eye(y_preds.shape[1])[y_true]
    # y_pred = np.array(y_preds)
    # # Avoid log(0) by clipping probabilities
    # y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # # Compute cross-entropy for each observation
    # ce = -np.sum(y_true * np.log(y_pred), axis=1)
    # # Return average cross-entropy
    # return np.mean(ce)


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = None)-> Tuple[float, float]:
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
    model.to(device)
    model.train()
    train_loss, train_acc, train_ce= 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        # Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss    # Accumulatively add up the loss per epoch 

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

       # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        # Calculate Cross entropy
        train_ce += cross_entropy_fn(y_true=y,
                                     y_preds=y_pred) 
        # train_ce += cross_entropy_fn(y_true=y.detach().numpy(),
        #                              y_preds=y_pred.detach().numpy()) 

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_ce = train_ce/ len(dataloader)
    return train_loss, train_acc, train_ce


def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               device: torch.device = None)-> Tuple[float, float]:
   
    """Test a PyTorch model for 1 epoch.

    Turns a target PyTorch model to eval mode and then
    runs forward pass on the test set

    Args:
        model: A PyTorch model to be used
        dataloader: A DataLoader instance for testing the model
        loss_fn: A PyTorch loss function to minimize.

    Returns:
        A tuple of test loss and test accuracy metrics.
        In the form (test_loss, test_accuracy)
    
  """
    test_loss, test_acc, test_ce = 0, 0, 0
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for X, y in dataloader:
            
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred = model(X)
                
            # Calculate loss (accumatively)
            test_loss += loss_fn(test_pred, y) 
            
            # Calculate accuracy
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            
            # Calculate Cross entropy
            test_ce += cross_entropy_fn(y_true=y,y_preds=test_pred) 

        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        test_ce /= len(dataloader)
        
        
    return test_loss, test_acc, test_ce
     
     
def train_test_loop(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          print_b: bool = True,
          Scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          early_stopping: EarlyStopping = None,
          device: torch.device = None
          ) -> Dict[str, List]:   
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
        train_loss, train_acc, train_ce = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device = device)
        
        test_loss, test_acc, test_ce = test_step(model=model, dataloader=test_dataloader,
                                        loss_fn=loss_fn,device=device)
        
        if early_stopping is not None:
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        
        # Adjust learning rate
        if Scheduler is not None:
            Scheduler.step()  

        

      # Print out what's happening
        if print_b:
            print(
                f"Epoch: {epoch+1} | "
                f"test_ce: {test_ce:.5f} | "
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


def save_model(model:torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.
  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),f=model_save_path)
  
## For dealing with Soft Labals
  
def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)  # Convert logits to log-probabilities
    loss = -torch.sum(soft_targets * log_probs, dim=1).mean()  # Compute soft cross-entropy
    return loss

def train_soft_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device = None)-> Tuple[float, float]:
    """Trains a PyTorch model for 1 epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps 

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        optimizer: A PyTorch optimizer to help minimize the loss function.

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy)
    
  """
    model.to(device)
    model.train()
    train_loss, train_acc= 0, 0

    for batch, (X, soft_label) in enumerate(dataloader):
        
        X, soft_label = X.to(device), soft_label.to(device)

        logits = model(X)

        # Calculate loss (per batch)
        loss = soft_cross_entropy(logits, soft_label)
        train_loss += loss    # Accumulatively add up the loss per epoch 

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()
        
        # Get accuracy 
        probabilities = torch.softmax(logits, dim=1)
        correct_predictions = (probabilities >= 0.5) * soft_label  # Match high probabilities to soft labels
 
    train_acc = correct_predictions.sum() / soft_label.sum()  
    train_loss = train_loss / len(dataloader)

    return train_loss, train_acc

def test_soft_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               device: torch.device = None)-> Tuple[float, float]:
   
    """Test a PyTorch model for 1 epoch.

    Turns a target PyTorch model to eval mode and then
    runs forward pass on the test set

    Args:
        model: A PyTorch model to be used
        dataloader: A DataLoader instance for testing the model
        loss_fn: A PyTorch loss function to minimize.

    Returns:
        A tuple of test loss and test accuracy metrics.
        In the form (test_loss, test_accuracy)
    
  """
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for X, soft_label in dataloader:
            
            X, soft_label = X.to(device), soft_label.to(device)

            # Forward pass
            logits_pred = model(X)
                
            # Calculate loss 
            loss = soft_cross_entropy(logits_pred, soft_label)
            test_loss += loss    # Accumulatively add up the loss per epoch 

            # Get accuracy 
            probs_pred = torch.softmax(logits_pred, dim=1)
            correct_predictions = (probs_pred >= 0.5) * soft_label  # Match high probabilities to soft labels
 
        # Divide total test loss by length of test dataloader (per batch)
        test_acc = correct_predictions.sum() / soft_label.sum()  
        test_loss = test_loss / len(dataloader)
            
    return test_loss, test_acc



def train_test_soft_loop(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          print_b: bool = True,
          Scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          early_stopping: EarlyStopping = None,
          device: torch.device = None
          ) -> Dict[str, List]:   
    """ Train test loop by epochs.

    Conduct train test loop 

    Args:
        model: A PyTorch model to be used
        train_dataloader: A DataLoader instance for trainig the model
        test_dataloader: A DataLoader instance for testinig the model
        optimizer: A PyTorch optimizer to help minimize the loss function.
        epochs: Number of epochs to run
        print_b: Boolean. When True the epochs and the test accuracy is printed. 
        device: device 


    Returns:
        A list of train loss, train accuracy metrics, test loss,
        test accuracy metrics.
        In the form (train_loss, train_accuracy,test_loss, test_accuracy)
    
  """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
                    
    for epoch in range(epochs):
        train_loss, train_acc = train_soft_step(model=model,
                                           dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           device = device)
        
        test_loss, test_acc = test_soft_step(model= model, 
                                             dataloader=test_dataloader,
                                             device=device)
        
        if early_stopping is not None:
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Adjust learning rate
        if Scheduler is not None:
            Scheduler.step()  

        # Print out what's happening
        if print_b:
            print(
                f"Epoch: {epoch+1} | "
                f"test_acc: {test_acc:.5f}"
            )

      # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        
    return results

  
def train_step_reg(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
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
        loss_fn: A PyTorch base loss function to minimize.
        reg_fn: regularized function with inputs, logits_main and logits_sub 
        optimizer: A PyTorch optimizer to help minimize the loss function.
        alpha: weight for the regularization term
        device: Torch device

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy)
    
  """
    model.train()
    train_loss, train_acc, train_ce= 0, 0, 0

    for batch, (X, y_main, y_sub) in enumerate(dataloader):
        X, y_main, y_sub = X.to(device), y_main.to(device), y_sub.to(device)
        y_predmain, y_predsub = model(X)

        # Optimizer zero grad
        optimizer.zero_grad()
        
        # Mask NaN values 
        valid_sub_mask = ~torch.isnan(y_sub) & (y_sub != -9223372036854775808)
        sub_y_clean = y_sub[valid_sub_mask].long() # remove NaNs
        y_predsub_clean = y_predsub[valid_sub_mask]
        y_predmain_clean = y_predmain[valid_sub_mask]

        # Calculate standard loss
        base_loss = loss_fn(y_predsub_clean, sub_y_clean)
        # add regularization term
        sbr_loss = reg_fn(logits_main = y_predmain_clean,
                          true_sublabels = sub_y_clean,
                          coef_lambda = coef_lambda)
        
        loss = base_loss + alpha * sbr_loss
        
        train_loss += loss    # Accumulatively add up the loss per epoch
        
        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

       # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_predsub_clean, dim=1), dim=1)
        train_acc += (y_pred_class == sub_y_clean).sum().item()/len(y_pred_class)

        # Calculate Cross entropy
        train_ce += engine.cross_entropy_fn(
            y_true=sub_y_clean.detach().numpy(),
            y_preds=y_predsub_clean.detach().numpy()).sum().item()/len(y_predsub)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_ce = train_ce/ len(dataloader)
    return train_loss, train_acc, train_ce


def test_step_reg(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
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
        loss_fn: A PyTorch loss function to minimize.
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
                
            # Mask NaNs
            valid_sub_mask = ~torch.isnan(y_sub) & (y_sub != -9223372036854775808)
            y_sub_clean = y_sub[valid_sub_mask].long() # remove NaNs
            test_predsub_clean = test_predsub[valid_sub_mask]
            test_predmain_clean = test_predmain[valid_sub_mask]

            # Calculate loss (accumatively)
            sbr_loss = reg_fn(logits_main = test_predmain_clean,
                              true_sublabels = y_sub_clean,
                             coef_lambda = coef_lambda)
            
            base_loss = loss_fn(test_predsub_clean, y_sub_clean)
            
            loss = base_loss + alpha * sbr_loss
                              
            test_loss += loss  
           
            # Calculate accuracy over subclasses
            test_acc += engine.accuracy_fn(y_true=y_sub_clean, y_pred=test_predsub_clean.argmax(dim=1))

            # Calculate Cross entropy
            test_ce += engine.cross_entropy_fn(y_true=y_sub_clean,y_preds=test_predsub_clean) 

        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        test_ce /= len(dataloader)
        
        
    return test_loss, test_acc, test_ce

def train_test_loop_reg(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          reg_fn,
          epochs: int,
          print_b: True ,
          alpha: float,
          device: torch.device = None,
          Scheduler: torch.optim.lr_scheduler._LRScheduler = None,
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
        train_loss, train_acc, train_ce = train_step_reg(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn = loss_fn,
                                           reg_fn = reg_fn,
                                           optimizer = optimizer,
                                           alpha = alpha,
                                           device = device,
                                           coef_lambda= coef_lambda)
        
        test_loss, test_acc, test_ce = test_step_reg(model=model, 
                                                    dataloader=test_dataloader,
                                                    loss_fn=loss_fn,
                                                    reg_fn= reg_fn,
                                                    alpha = alpha,
                                                    device =device,
                                                    coef_lambda= coef_lambda)
        
      # Adjust learning rate
        if Scheduler is not None:
            Scheduler.step()  

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