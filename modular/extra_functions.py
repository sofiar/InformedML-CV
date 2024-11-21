"""
Contains extra functions to simulate data and save models.
"""
from pathlib import Path
import numpy as np
import cv2
import torch


def create_blank_image(height:float, width:float):
    """Create a blank image.

    Args:
      hight: as number of pixels.
      width: as number of pixels.
      
    Returns:
      a numpy array
    
    """
    return np.zeros((height, width), dtype=np.uint8)

# Function to draw a circle on a binary image
def draw_circle(image_size, radius):
    """draw circle.

    Simulate circle images

    Args:
        image_size: as number of pixels
        radius: radius of the circle
        

    Returns:
        A numpy.ndarray
    
  """
    image = create_blank_image(image_size, image_size)
    center = (image_size // 2, image_size // 2)
    cv2.circle(image, center, radius, (1), -1)  # Draw filled circle
    
    ## Add rotation
    center = (image_size // 2, image_size // 2)
    angle = np.random.uniform(0, 180)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image_size, image_size), 
                                   flags=cv2.INTER_CUBIC)
    return rotated_image
    
# Function to draw a square on a binary image
def draw_square(image_size, square_size):
    """draw square.

    Simulate square images

    Args:
        image_size: as number of pixels
        square_size: length of the square
        

    Returns:
        A numpy.ndarray
    
  """
    image = create_blank_image(image_size, image_size)
    top_left = (image_size // 2 - square_size // 2, 
                image_size // 2 - square_size // 2)
    bottom_right = (image_size // 2 + square_size // 2, 
                    image_size // 2 + square_size // 2)
    cv2.rectangle(image, top_left, bottom_right,(1),-1)  
      
    ## Add rotation
    center = (image_size // 2, image_size // 2)
    angle =np.random.uniform(0, 180)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image_size, image_size), 
                                   flags=cv2.INTER_CUBIC)
    return rotated_image
  
# Function to draw a triangle on a binary image
def draw_triangle(image_size, triangle_size):
    """draw triangle.

    Simulate triangle images

    Args:
        image_size: as number of pixels
        square_size: length of the circle
        

    Returns:
        A numpy.ndarray
    
  """
    image = create_blank_image(image_size, image_size)
    
   # Calculate the vertices for a centered triangle
    center_x, center_y = image_size // 2, image_size // 2
    vertices = np.array([[
        (center_x, center_y - triangle_size // 2),  # Top vertex
        (center_x - triangle_size // 2, center_y + triangle_size // 2),  # Bottom-left
        (center_x + triangle_size // 2, center_y + triangle_size // 2)  # Bottom-right
    ]], dtype=np.int32)
    cv2.fillPoly(image, vertices,1)
    
    ## Add rotation
    center = (image_size // 2, image_size // 2)
    angle =np.random.uniform(0, 180)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image_size, image_size), 
                                   flags=cv2.INTER_CUBIC)
    return rotated_image  
  
def add_gaussian_noise(image, mean=0, var=0.15):
    """Add Gaussian noise to image.

    Args:
      image: numpy.ndarray
      mean: 0.
      var: variance for the gaussin noise

    """
    row, col = image.shape
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, 0, 1)  # Ensure values remain within 0-1
    return noisy_image

