# Introduction

In this project, we developed a neural network model using the Keras API to predict whether an individual's income exceeds $50,000 per year, using the Adult dataset from the UCI Machine Learning Repository. The dataset contains 14 features and 1 target variable, which represents income classification. Below is a summary of the dataset characteristics:

| Variable Name    | Role    | Type         |
| :--------------- | :------ | :----------- |
| age              | Feature | Integer      |
| workclass        | Feature | Categorical  |
| fnlwgt           | Feature | Integer      |
| education        | Feature | Categorical  |
| education-num    | Feature | Integer      |
| marital-status   | Feature | Categorical  |
| occupation       | Feature | Categorical  |
| relationship     | Feature | Categorical  |
| race             | Feature | Categorical  |
| sex              | Feature | Binary       |
| capital-gain     | Feature | Integer      |
| capital-loss     | Feature | Integer      |
| hours-per-week   | Feature | Integer      |
| native-country   | Feature | Categorical  |
| income           | Target  | Binary       |

# Changes Added

Below is the additional changes made to the original code:
- Instead of using read_csv, we use the ucimlrepo package in order to fetch the dataset directly.
- We also add some additional hyperparameter for testing and fine-tuning to see better results.

# 1. Preprocessing

As part of the preprocessing pipeline, we performed the following steps:

- **Categorical Encoding**: Converted all categorical values into numerical representations.

- **Standardization and Normalization**: Applied standardization to ensure all numerical features had a mean of 0 and a standard deviation of 1, and normalized values to fit within a specific range.

- **Handling Missing Data**: Removed all rows containing null values to ensure a clean dataset for model training.

# 2. Model Preparation

We used the Keras API to train the neural network. The original model was trained using the following hyperparameters:

- Number of neurons per hidden layer: Initially set to 5 neurons for all hidden layers.

Here are some graph that shows the **accuracy vs. epoch** and **loss vs. epoch** for model configuration without any hyperparameter tuning.

To see all of the images, go to
<!-- insert link here -->

From the results, it became clear that the models exhibited overfitting. Training accuracy was significantly higher than validation accuracy, indicating the need for hyperparameter fine-tuning. We will be using the **ReLu activation function** as it gives as the best result.

# 3. Fine-Tuning the Model

To address the overfitting and improve model performance, we experimented with several hyperparameter adjustments. Below are the key changes made during the fine-tuning process:

## Adjusting the Number of Neurons
Initially, we set the number of neurons in each hidden layer to 5. To explore optimal configurations, we tested various neuron counts in the hidden layers. After comparing the results, we found that 30 neurons per hidden layer achieved the best performance.

<!-- insert image of all neuron training -->

## Regularization and Early Stopping
To further mitigate overfitting, we introduced L2 regularization with a factor of 0.05 to simplify the model and constrain overfitting. Additionally, we applied early stopping, which halts training when the model’s performance stops improving on the validation set, preventing unnecessary training epochs.

## Learning Rate Optimization
Next, we fine-tuned the learning rate. After testing different rates, we determined that a learning rate of 1e-4 provided the most optimal results. A rate of 1e-5 led to underfitting, while a rate of 1e-3 or higher caused the model to overfit more rapidly.

## Final Model Architecture
The final architecture used the ReLU activation function for the hidden layers and 30 neurons per layer. The best model configuration is shown below, along with the graph of accuracy vs. epoch and loss vs. epoch for the tuned model.

<!-- insert image of optimal architecture and graph of accuracy vs epoch here -->

## Final Hyperparameter Tuning
The table below summarizes the final hyperparameters after fine-tuning:

| Hyperparameter            | Value       |
| :------------------------ | :---------- |
| Number of neurons          | 30          |
| Activation function        | ReLU        |
| Learning rate              | 1e-4        |
| L2 regularization          | 0.05        |
| Early stopping patience    | 10 epochs   |


# 4. Result

After applying all the optimizations, the final model achieved an accuracy of 94% on the Adult dataset. This performance reflects a significant improvement from the initial model, and the model generalizes well to unseen data.