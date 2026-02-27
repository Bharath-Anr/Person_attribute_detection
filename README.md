Classify person attributes like long, short, no **beard and hair**. **eyewear detection, gender, hat detection.**

This project implements a multi-head deep learning model that predicts multiple facial attributes from a single image using a shared backbone network.

## **Dataset:**
I used the **CelebA dataset** which has 200k high quality images of faces of people.

### link: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data
##
Backbone: ResNet-18

Framework: PyTorch

Training Strategy: Multi-task learning
##
The model predicts:

  Hair type (3 classes)
  
  Beard type (3 classes)
  
  Eyewear (2 classes)
  
  Gender (2 classes)
  
  Hat (2 classes)
  
##
Instead of training separate models for each attribute, this project uses:
  A shared ResNet18 backbone, Multiple independent classification heads, Joint training using summed loss
  
  This improves efficiency and generalization.
##
**Drive link for the **model trained on 10k images** of celebA dataset**

https://drive.google.com/file/d/1pT-wrV93Qeja1jf5kbRTyA1aZ5YnrbK0/view?usp=drive_link

## **1️.model.py:**

  Defines the neural network architecture.
  
  **Key Components**
  
  Loads pretrained ResNet18
  
  Removes original ImageNet classifier
  
  Adds 5 attribute-specific classification heads
  
  Returns predictions as a dictionary
  

## **2. celeba_dataset.py:**

  Custom PyTorch Dataset class.
  
  Responsibilities
  
  Reads CSV file
  
  Loads images from directory
  
  Applies transformations
  
  Returns:
  
  Image tensor
  Dictionary of labels
    
  Expected CSV Format:
  
    image_id,hair,beard,eyewear,gender,hat 
    000001.jpg,1,0,0,1,0

   
## **3. train_multihead.py:**

  Handles complete training pipeline:
  
  Device configuration
  
  Image transforms
  
  Dataset loading
  
  Model training
  
  Validation
  
  Model saving


## **4.batch_sort_predictions.py :**

  Purpose:-
  
  This script performs batch inference on a folder of images and:
  
  Loads the trained multi-head model
  
  Predicts all attributes for each image
  
  Prints both human-readable and raw outputs
    
  Automatically sorts images into predicted folders
   Measures pure forward-pass inference time
    
