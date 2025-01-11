Task 1

Using D1 from the CIFAR-10 dataset, train a model  with a Learning with Prototypes (LwP) classifier. Sequentially predict the labels of datasets D2 to D10 using the previous model and update it (e.g., ). The primary goals are:

Ensure high accuracy on the ith held-out dataset .

Maintain performance on earlier datasets  when moving to .

Task 2

Starting with  from Task 1, sequentially train models  using datasets D11 to D20. Unlike Task 1, these datasets have different input distributions. The primary goals are:

Achieve good accuracy on the ith held-out dataset .

Maintain performance on earlier datasets  when transitioning to .

Structure

Code Overview:

The code is organized into several sections:

Library Imports: All necessary libraries like NumPy, PyTorch, and Sklearn.

Data Preparation: Loading CIFAR-10 datasets, splitting labeled and unlabeled data.

Model Definition: Implementation of the LwP classifier.

Training Loop: Sequentially training models on datasets (D1 to D10 for Task 1, D11 to D20 for Task 2).

Evaluation: Calculating accuracy on held-out datasets.

File Organization:

task1.ipynb: Notebook for Task 1.

task2.ipynb: Notebook for Task 2.

utils.py: Helper functions for data loading and preprocessing (if provided).

README.md: This file.

Strategy

Common Approach

Feature Extraction:

Use a pre-trained neural network (e.g., ResNet) to extract feature representations from raw images.

Evaluation:

After training each model , evaluate it on  and all previous held-out datasets .

Report accuracy in a matrix format (10x10 for Task 1, 10x20 for Task 2).

Task-Specific Strategies

Task 1

Train  on D1 using labeled data.

Use the predictions of  on D2 as pseudo-labels to train .

Repeat the process iteratively for D3 to D10.

Task 2

Fine-tune  on D11 with domain adaptation techniques (e.g., reweighting samples or adding domain-specific parameters).

Continue fine-tuning sequentially on D12 through D20, considering the distribution differences.

How to Run the Code

Clone the repository:

git clone <repo_url>
cd mini-project-2

Install required packages:

pip install -r requirements.txt

Run the notebooks:

Task 1:

jupyter notebook task1.ipynb

Task 2:

jupyter notebook task2.ipynb



