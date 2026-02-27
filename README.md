# Person_attribute_detection
Classify person attributes like long, short, no beard / hair. eyewear detection, gender, hat detection.

This project implements a multi-head deep learning model that predicts multiple facial attributes from a single image using a shared backbone network.

The model predicts:
Hair type (3 classes)
Beard type (3 classes)
Eyewear (2 classes)
Gender (2 classes)
Hat (2 classes)
The backbone used is ResNet-18, with separate classification heads for each attribute.
