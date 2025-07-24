# ##############################################################################
#                                                                              #
#                              Prior shift simulation                          #
#                                                                              # 
################################################################################

import numpy as np 
import torch 
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score

import modular.samples_setup as cs
from modular import model_builder
from modular import engine


################################################################################
#                           Functions definitions                              #
################################################################################

def predict_sample(model, image):

    with torch.no_grad():
        out = model(image)  
        y_pred = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)

    return y_pred, probs


def predict_sample_adjusted(model, image, priors_new, priors_train):
    
    with torch.no_grad():
        out = model(image)   
        out_adj = out.clone()
        
        for index in range(len(out[0])):
            out_adj[0][index] = (
                out[0][index]
                + np.log(priors_new[index]) 
                - np.log(priors_train[index])
            )
            
        y_pred = out_adj.argmax(dim=1)
        probs = F.softmax(out_adj, dim=1)
        
    return y_pred, probs 


################################################################################
#                              Perform simulations                             #
################################################################################


SEED = 1222
BATCH_SIZE = 50
EPOCHS = 40
NREPS = 30 
NOISE_PROP = 1
VAR = 0.1
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# To reduce variability when re-running 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set number of training samples
nsamples_train = [
    [2000] * 3, # equal proportion
    [1000, 1500, 2000], # unbalanced
    [300,1500,1500]  # one class under-represented
]

# Set number of new samples
nsamples_new = [
    [2000] * 3, # equal proportion
    [1000, 1500, 2000], # unbalanced
    [300,1500,1500]  # one class under-represented
]
# Objects to save results
accuracy_no_adj = []
accuracy_adj = []

ce_no_adj = []
ce_adj = []

precision_no_adj = []
precision_adj = []

recall_no_adj = []
recall_adj = []

all_train_sample = []
all_new_sample = []
n_rep = []

for rep in range(NREPS):
    
    for ind, train_sample in enumerate(nsamples_train):
        
        curr_new_samples = nsamples_new#[:ind] + nsamples_new[ind+1:] 
        indx_new_sample = [0,1,2]#[:ind]+ [0,1,2][ind+1:]

        
        # 1. Simulate training sample
        
        output = cs.generate_sample(
            n=train_sample, 
            noise_prop=NOISE_PROP, 
            var=VAR, 
            Nclass=3 
        )
        images, labels = (output['images'], output['labels'])
        
        # Generate dataloaders
        
        train_dataset, test_dataset = cs.generate_dataset(images, labels)

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size= 32, 
            shuffle=False
            )
                                    
        test_dataloader = DataLoader(
            test_dataset,
            batch_size= 32,
            shuffle=False
            )
        
        # Priors in Training 
        labels_train = train_dataset.tensors[1].numpy()
        priors_original = np.unique(labels_train,return_counts=True)[1]/len(labels_train)
        
        # 2. Define model
        
        my_model =  model_builder.TVGG(
        input_shape=1, 
        hidden_units=10, 
        output_shape=3
        )

        # Loss Function
        loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.SGD(params=my_model.parameters(), lr=0.05)
        
        # 3. Train test loop 

        output = engine.train_test_loop(
            my_model,
            train_dataloader,
            test_dataloader, 
            optimizer, 
            loss_fn,
            EPOCHS,
            print_b=False
        )

        # 4. Simulate new sample
        
        for indx, new_sample in enumerate(curr_new_samples):
            
            n_rep.append(rep)
            all_train_sample.append(train_sample)
            all_new_sample.append(new_sample)
                
            output_new = cs.generate_sample(
                n=new_sample, 
                noise_prop=NOISE_PROP, 
                var=VAR, 
                Nclass=3 
            )
        
            images_new, labels_new= (output_new['images'], output_new['labels'])

            # Generate dataloaders
            data_new, test_new = cs.generate_dataset(
                images_new, 
                labels_new, 
                test_size=0.05,
                seed=999
            )

            new_dataloader = DataLoader(
                data_new, 
                batch_size= 1, 
                shuffle = False
            )
            
            labels_new = data_new.tensors[1].numpy()
            priors_updated = np.unique(labels_new,return_counts=True)[1]/len(labels_new)

            # 5. Assess original model
            
            predictions = []
            true_labels = []
            ce = 0

            for imag, label  in new_dataloader:
                true_labels.append(label.numpy()[0])
                predict_out = predict_sample(model=my_model,image=imag) 
                predictions.append(predict_out[0].numpy()[0])
                ce = ce -np.log(predict_out[1][0][label.numpy()[0]])
                    
            ce_no_adj.append(ce.numpy()/len(new_dataloader))
            accuracy_no_adj.append(accuracy_score(predictions, true_labels))
            recall_no_adj.append(recall_score(predictions, true_labels, average=None))
            precision_no_adj.append(
                precision_score(predictions, true_labels,average=None)
            )      
            
            # 6. Adjust prior shift
            
            predictions_adj = []
            true_labels = []
            ce_adjt = 0

            for imag, label  in new_dataloader:
                true_labels.append(label.numpy()[0])
                predict_adj_out = predict_sample_adjusted(
                    model=my_model,
                    image=imag, 
                    priors_new=priors_updated,
                    priors_train=priors_original
                    )
                predictions_adj.append(predict_adj_out[0].numpy()[0])
                ce_adjt = ce_adjt -np.log(predict_adj_out[1][0][label.numpy()[0]])

            ce_adj.append(ce_adjt.numpy()/len(new_dataloader))
            accuracy_adj.append(accuracy_score(predictions_adj, true_labels))
            recall_adj.append(recall_score(predictions_adj, true_labels, average=None))
            precision_adj.append(
                precision_score(predictions_adj, true_labels,average=None)
            )     
        

################################################################################
#                               Create Dataframe                               #
################################################################################


data = list(zip(
    n_rep,
    all_train_sample,
    all_new_sample,
    accuracy_no_adj,
    accuracy_adj,
    ce_no_adj,
     ce_adj
    ))

df = pd.DataFrame(
    data, 
    columns=[
        'Nrep',
        'Sample_train', 
        'Sample_new', 
        'Accuracy',
        'Adj_Accuracy',
        'C-E','Adj_ce'
        ]
    )

# Create df with recall and precision per class
df_recall = pd.DataFrame(
    recall_no_adj,
    columns=['Recall_circle', 'Recall_square', 'Recall_triangle']
)

df_recall_adj = pd.DataFrame(
    recall_adj, 
    columns=['Recall_adj_circle', 'Recall_adj_square', 'Recall_adj_triangle']
)

df_precision = pd.DataFrame(
    precision_no_adj, 
    columns=['Precision_circle', 'Precision_square', 'Precision_triangle']
)

df_precision_adj = pd.DataFrame(
    precision_adj, 
    columns=['Precision_adj_circle', 'Precision_adj_square', 'Precision_adj_triangle']
)

final_df = pd.concat(
    [df, df_recall,df_recall_adj,df_precision,df_precision_adj], 
    axis=1
    )


################################################################################
#                                Save results                                  #
################################################################################    


env_vars = {'final_df':final_df }

# Save variables using joblib 
where_to_save = (
    '/home/sofia/Candu_postdoc/InformedMlCv/Environments/'
    'prior_shift.pkl'
)
torch.save(env_vars, where_to_save)

print('Done!')  
