"""
Contains functionality for genenrating and setting up samples for 
image classification data.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import modular.extra_functions as ef

def generate_sample(
    noise_prop, 
    seed = 999,
    size = 28, 
    var = 0.15,
    noise_indx = False, 
    order = None,
    n=None, 
    Nclass = 2
    ):
    
    """Generates samples.

     Args:
        n: vector number of images of each class to be generated.
        in the order: circles, squares and triangles(this last can be included 
        or not). 
        noise_prop: proportion of the data with noise
        seed: seed to perform the simulation. Default = 999.
        size: size of the image as number of pixels.
        var: variance for the Gaussian Error
        order: np.array of 0(circles) and 1(squares), representing the sequence
        of circles and squares to  simulate. If n is given order do not apply!
        Nclass: number of classes. Nclass must be 2 o 3. Default = 2. 

    Returns:
        A Dictionary of Arrays of images and labels simulated.
        In the form (images, labels, noisy index)
    
  """
    
    if(n is not None and order is not None):
        print(
            "Error: n and order cannot be input at the same time" +
            "please select only one method"
        )
        
    if(Nclass not in [2,3]):
        print("Error: Nclass must be 2 or 3")      

    else: 
        dataset_circles = []
        dataset_squares = []

        if(n is not None):
            if (len(n)!=Nclass):
                 print("Error: please provide n of length Nclass")
            else:
                ntot = sum(n)
                n_cr = n[0]
                n_sq = n[1]
                
                for _ in range(n_cr):
                    radius = np.random.randint(2, size // 2)  
                    image = ef.draw_circle(size, radius)
                    dataset_circles.append(image)
                    
                for _ in range(n_sq):
                    len_size = np.random.randint(2, size // 2)  
                    image = ef.draw_square(size, len_size)
                    dataset_squares.append(image)
                        
                if (Nclass==3):
                    n_tr = n[2]
                    dataset_triangles = []    
                    for _ in range(n_tr):
                        t_size = np.random.randint(2, size // 2)  
                        image = ef.draw_triangle(size, t_size)
                        dataset_triangles.append(image)    
                
            # # Add noise to some of them
            # n_noisy = int(n * noise_prop)
            # random.seed(seed)
            # noisy_indices_c = np.random.choice(n, n_noisy, replace=False)
            # noisy_indices_s = np.random.choice(n, n_noisy, replace=False)    
                
        if(order is not None):
            np.random.seed(seed)
            if(Nclass==2):
                n_sq = np.sum(order == 1)
                n_cr = np.sum(order == 0)
                ntot = n_cr + n_sq
                
                for _ in range(n_sq):
                    len_size = np.random.randint(2, size // 2)  # Random len between 2 and half the image size
                    image = ef.draw_square(size, len_size)
                    dataset_squares.append(image)
                
                for _ in range(n_cr):    
                    radius = np.random.randint(2, size // 2)  # Random radius between 2 and half the image size
                    image = ef.draw_circle(size, radius)
                    dataset_circles.append(image)
            
            if(Nclass==3):
                dataset_triangles = []    
                n_sq = np.sum(order == 1)
                n_cr = np.sum(order == 0)
                n_tr = np.sum(order == 2)
                ntot = n_cr + n_sq + n_tr
            
                for _ in range(n_sq):
                    len_size = np.random.randint(2, size // 2)  # Random len between 2 and half the image size
                    image = ef.draw_square(size, len_size)
                    dataset_squares.append(image)
            
                for _ in range(n_cr):    
                    radius = np.random.randint(2, size // 2)  # Random radius between 2 and half the image size
                    image = ef.draw_circle(size, radius)
                    dataset_circles.append(image)
            
                for _ in range(n_tr):
                    t_size = np.random.randint(2, size // 2)  # Random len between 2 and half the image size
                    image = ef.draw_triangle(size, t_size)
                    dataset_triangles.append(image)
            
        # Add noise to some of them
        n_noisy_cr = int(n_cr * noise_prop)
        n_noisy_sq = int(n_sq * noise_prop)
        np.random.seed(seed)
        noisy_indices_c = np.random.choice(n_cr, n_noisy_cr, replace=False)
        noisy_indices_s = np.random.choice(n_sq, n_noisy_sq, replace=False)       
              
        for i in noisy_indices_c:
            dataset_circles[i] = ef.add_gaussian_noise(
                dataset_circles[i],
                var=var
            )
            
        for i in noisy_indices_s:
            dataset_squares[i] = ef.add_gaussian_noise(
                dataset_squares[i],
                var=var
            )
            
        # Create labels
        circle_labels = np.full(n_cr, 0)
        square_labels = np.full(n_sq, 1)
        
        # # Combine datasets
        all_images = np.concatenate((dataset_circles, dataset_squares))
        all_labels = np.concatenate((circle_labels, square_labels))
        
        if (Nclass==3):
            np.random.seed(seed)
            n_noisy_tr = int(n_tr * noise_prop)
            noisy_indices_t = np.random.choice(n_tr, n_noisy_tr, replace=False)       
            
            for i in noisy_indices_t:
                dataset_triangles[i] = ef.add_gaussian_noise(
                    dataset_triangles[i],
                    var=var
                )
            # Create labels
            triangle_labels = np.full(n_tr, 2)
            
            all_images = np.concatenate((all_images, dataset_triangles))
            all_labels = np.concatenate((all_labels, triangle_labels))
        
        if (order is not None):
            indx_sq = np.where(order==1)
            indx_cr = np.where(order==0)
            all_images[indx_sq] = dataset_squares
            all_images[indx_cr] = dataset_circles
            all_labels[indx_sq] = square_labels
            all_labels[indx_cr] = circle_labels
            
            if (Nclass==3):
                indx_tr = np.where(order==2)
                all_images[indx_tr] = dataset_triangles
                all_labels[indx_tr] = triangle_labels
        else:    
            # Shuffle the dataset
            indices = np.arange(len(all_images))
            np.random.seed(seed)
            np.random.shuffle(indices)
            all_images = all_images[indices]
            all_labels = all_labels[indices]
            noisy_indx  = np.concatenate((noisy_indices_c,ntot + noisy_indices_s))
            noisy_indx = np.nonzero(np.isin(indices, noisy_indx))[0]

        
        # recover noisy indices
        if noise_indx:
            noisy_indx  = np.concatenate(
                (noisy_indices_c,
                 ntot + noisy_indices_s)
                )
                
            output = {
                "images": all_images,
                "labels": all_labels,
                "indx_noisy_data": noisy_indx
                }
        else:
            output = {
                "images": all_images,
                "labels": all_labels
                }
            
        return output
        

def generate_dataset(all_images, all_labels, test_size=0.2,seed = 999):

    """Generate dataset.

     Args:
        all_images: A array with images generated from `generate_sample`
        all_labels: A array of labels corresponding to the images,
        generated from`generate_sample`
        seed: seed to perform the simulation. Default = 999.

        
    Returns:
        A tuple of TensorDatasets of train and test data sets.
        In the form (train_dataset, test_dataset)
    
  """
    
    X = torch.from_numpy(all_images).type(torch.float)
    Y = torch.from_numpy(all_labels).type(torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=seed
    ) 

    # Add channel at dimension 1 (greyscale)
    X_train = X_train.unsqueeze(1)  
    X_test = X_test.unsqueeze(1)
        
    train_dataset = torch.utils.data.TensorDataset(X_train,y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test,y_test)

    return train_dataset, test_dataset