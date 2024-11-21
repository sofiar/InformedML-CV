
"""
Contains functions for training and testing a PyTorch model.
"""
import torch 
from timeit import default_timer as timer 


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer)-> Tuple[float, float]:
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
    train_loss(), train_acc()= 0, 0
  
    for batch, (X, y) in enumerate(dataloader):
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch 

        # 3. Optimizer zero grad
        optimizer.zero_grad()
            
        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module)-> Tuple[float, float]:
    
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
    model.eval()
        
    with torch.inference_mode():
        for X, y in dataloader:
            #forward pass
            test_pred = model(X)
                
            # calculate loss (accumatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # Calculate accuracy
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        
    return test_loss, test_acc

            
            



#                 ## Print out what's happening
#             #print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

#         return(test_acc, test_loss,test_pred.argmax(dim=1))

# def train_test_loop(epochs,train_dataloader,test_dataloader, 
#                     model,optimizer,loss_fn): 
    
#     # Create training and testing loop
#     for epoch in range(epochs):
#         train_loss = 0
#         # Add a loop to loop through training batches
#         for batch, (X, y) in enumerate(train_dataloader):
#             model.train() 

#             # 1. Forward pass
#             y_pred = model(X)

#             # 2. Calculate loss (per batch)
#             loss = loss_fn(y_pred, y)
#             train_loss += loss # accumulatively add up the loss per epoch 

#             # 3. Optimizer zero grad
#             optimizer.zero_grad()

#             # 4. Loss backward
#             loss.backward()

#             # 5. Optimizer step
#             optimizer.step()

                
#         # Divide total train loss by length of train dataloader (average loss per batch per epoch)
#         train_loss /= len(train_dataloader)
            
#         ### Testing
#         # Setup variables for accumulatively adding up loss and accuracy 
#         test_loss, test_acc = 0, 0 
#         model.eval()
        
#         with torch.inference_mode():
#             for X, y in test_dataloader:
#                 # 1. Forward pass
#                 test_pred = model(X)
                
#                 # 2. Calculate loss (accumatively)
#                 test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

#                 # 3. Calculate accuracy (preds need to be same as y_true)
#                 test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
                
         
#                 # Calculations on test metrics need to happen inside torch.inference_mode()
#                 # Divide total test loss by length of test dataloader (per batch)
#             test_loss /= len(test_dataloader)
#             test_acc /= len(test_dataloader)



#                 ## Print out what's happening
#             #print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

#         return(test_acc, test_loss,test_pred.argmax(dim=1))