## cGAN Project

This project demonstrates the implementation of a Conditional Generative Adversarial Network (cGAN) using TensorFlow and Keras. The cGAN model is trained on the Fashion MNIST dataset to generate images conditioned on specific labels.

### Project Overview

The project is divided into several key steps:
1. **Setup and Data Preparation**: Importing necessary libraries, setting up the GPU configuration, and preparing the Fashion MNIST dataset by normalizing and reshaping the images, and converting labels to one-hot vectors.
2. **Building the Generator**: Constructing the generator model, which takes a noise vector and a label as input and generates an image corresponding to the label.
3. **Building the Discriminator**: Constructing the discriminator model, which takes an image and its corresponding label as input and outputs a scalar value representing the authenticity of the image.
4. **Compiling and Training the Models**: Setting up the loss functions, optimizers, and the training loop to simultaneously train the generator and discriminator.

### Key Features

- **Generator Model**: Utilizes Dense, Reshape, and Conv2DTranspose layers to generate 28x28 grayscale images.
- **Discriminator Model**: Utilizes Conv2D and Dense layers to classify images as real or fake.
- **Conditional GAN**: Implements conditioning by combining the noise vector with the label information.
- **GPU Support**: Configured to leverage GPU for accelerated training.

### How to Run

1. Clone the repository:
   ```bash
   git clone <github.com/MMansour97/CGAN.git>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook `Übung_cGAN.ipynb` and run the cells step by step to train the cGAN model.

### Results

After training, the cGAN model generates realistic images corresponding to the specified labels. The generated images improve in quality as training progresses.

### Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib

### Acknowledgements

This project is inspired by the Conditional GANs and their applications in generating images conditioned on labels. The Fashion MNIST dataset is used for training and evaluation.

---
Note: This project was an exercise for the students at my Hochschule THD. They were required to try to solve the project, and this is the final result.
