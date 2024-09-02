# Neural Network Regression Model with Hyperparameter Tuning

This project implements a neural network model for regression using Keras. The goal is to predict a target variable (`PAC`) from a given dataset, optimize the model by testing various hyperparameter configurations, and evaluate its performance using metrics like R² and Mean Absolute Percentage Error (MAPE).

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Feature Selection](#feature-selection)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview

This project involves the following key steps:

1. **Data Preprocessing:** Load and preprocess the dataset, including standardization.
2. **Model Initialization:** Define a sequential neural network model with adjustable parameters.
3. **Hyperparameter Tuning:** Evaluate different configurations of the neural network, varying the number of neurons, hidden layers, batch size, and epochs.
4. **Performance Evaluation:** Calculate R² and MAPE for each configuration and identify the optimal hyperparameters.
5. **Feature Selection:** Identify the most correlated features with the target variable and rebuild the model using these selected features.
6. **Final Evaluation:** Evaluate the optimized model using the selected features.

## Installation

Ensure that you have Python installed along with the following libraries:

- Pandas
- Scikit-learn
- Keras
- TensorFlow (backend for Keras)

You can install these libraries using pip:

```bash
pip install pandas scikit-learn keras tensorflow
```

## Usage

To run the project, follow these steps:

1. **Load the Dataset:** Replace the file path with your own dataset.
2. **Run the Model:** Execute the script to train and evaluate the neural network model.
3. **View Results:** The script will output the R², MAPE, and the best hyperparameter configuration.


## Dataset

The dataset used in this project should be a tab-separated text file containing multiple features and the target variable (`PAC`). The first step of the script loads this dataset and splits it into training and testing sets.

## Model Architecture

The neural network is built using the Keras Sequential API. The architecture can be adjusted as follows:

- **Input Layer:** Based on the number of features in the dataset.
- **Hidden Layers:** Configurable number of hidden layers and neurons per layer.
- **Output Layer:** A single neuron with a linear activation function for regression.

The model is compiled using Mean Squared Error (MSE) as the loss function and the Adam optimizer.

## Hyperparameter Tuning

The project includes a loop to test various configurations:

- **Neurons:** [32, 64, 128]
- **Hidden Layers:** [1, 2, 3]
- **Batch Size:** [16, 32, 64]
- **Epochs:** [50, 100, 150]

Each configuration is evaluated on the test set, and the results are stored for comparison.

## Feature Selection

The script calculates the correlation matrix of the dataset to identify the most correlated features with the target variable (`PAC`). The top correlated features are selected to rebuild and evaluate the model with potentially improved performance.

## Results

The results include the following:

- **R² Score:** Coefficient of determination to measure the goodness of fit.
- **MAPE:** Mean Absolute Percentage Error to measure prediction accuracy.
- **Training Time:** Time taken to train the model for each configuration.

The optimal configuration with the best R² and lowest MAPE is highlighted in the final output.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request.
