Enchanted Wings: Marvels of Butterfly Species
Project Title:
Transfer Learning-Based Butterfly Image Classification for Biodiversity Monitoring
Overview:
This project aims to classify 75 different species of butterflies using a Convolutional Neural Network (CNN) architecture — EfficientNetB0 — through transfer learning. It is designed to support biodiversity monitoring, ecological research, and educational outreach by providing a fast, accurate image-based classification tool.
Objectives:
- Use pre-trained CNN (EfficientNetB0) for butterfly species classification.
- Achieve high accuracy with limited training data.
- Enable real-time classification for use in conservation, research,and citizen science.
Model Details:

  Architecture: EfficientNetB0 (Pre-trained on ImageNet)
  Input Image Size: 224×224 pixels
  Classes: 75 butterfly species
  Loss Function: Categorical Crossentropy
  Optimizer: Adam
  Epochs: 25
  Accuracy:
  Training: 96.2%
  Validation: 93.8%
  Test: 92.5%
  Inference Time: ~0.12 seconds/image
Dataset:

- Source: [Kaggle / Public Butterfly Dataset]
- Total Images: 6499
- Classes: 75
- Pre-split into: train/, valid/, and test/
Technologies Used:

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Flask (for deployment)
- Kaggle (for training & testing)
- HTML/CSS (for web UI)
