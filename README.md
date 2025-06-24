# Machine-learning-model-training-toolkit
A MATLAB-based toolkit for training machine learning models
Installation
Save the MachineLearningModelLearner.mlappinstall file to a folder included in your MATLAB path.
Open MATLAB and navigate to the APPS tab on the top menu bar.

Click Install App (top-right corner), then browse to the .mlappinstall file, select it, and click Open.

MATLAB will install the app and add it to the APPS page for easy access.

⚠️ Dependencies:
Please ensure the following MATLAB toolboxes are installed for full functionality:

Deep Learning Toolbox

Statistics and Machine Learning Toolbox

To install them:
Go to Home > Add-Ons > Get Add-Ons, then search and install both toolboxes from the Add-On Explorer.

Getting Started
Launch the app by clicking its icon in the APPS panel.

Prepare your dataset as an Excel file (.xlsx):

Input features should be arranged in the initial columns.

Output variables should be placed in the final columns.

Upon launch, the app interface will prompt you to:

Choose between Supervised and Unsupervised learning.

If Supervised, further select either Regression or Classification.

📊 Model Training
✅ Supervised Learning
Regression
Required inputs:

Excel file name

Number of output dimensions

Test data ratio (value between 0 and 1)

Validation methods:

Hold-out validation (recommended for large datasets)

K-fold cross validation (recommended for small datasets)

Output metrics:

R², MSE, MAE, RMSE

Classification
No need to specify the number of output variables (assumed 1D)

Same validation options as regression

Output metrics:

Confusion matrix

Accuracy, F1 score, Precision, Recall
