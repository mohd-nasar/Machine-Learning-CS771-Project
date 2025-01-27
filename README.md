# Machine Learning Project
This repository contains the implementation of a machine learning project for the CS771 course at IIT Kanpur, under the guidance of Professor Piyush Rai (Aug 2024 - Dec 2024).

## MiniProject1 - Emoticon Classifier

This project involves building and evaluating several machine learning models on different datasets, including text sequences, emoticons, and feature-based data. The models range from deep learning techniques (LSTM) to traditional machine learning classifiers (Random Forest, Logistic Regression). The models are trained on varying percentages of data and their performance is evaluated using validation accuracy. The final predictions are saved as text files for further analysis.

### Datasets
1. **Emoticon Dataset**: Contains emoticon data along with labels for classification. The emoticons are cleaned and preprocessed, and a deep learning model (LSTM) is used to extract features which are then classified using logistic regression.
   
2. **Feature Dataset**: Features are loaded from `.npz` files and then used to train a Random Forest classifier. The accuracy of training and validation data is evaluated.

3. **Text Sequence Dataset**: This dataset contains sequences of characters, and preprocessing steps are applied to remove substrings and encode characters. An LSTM model is used to predict labels from the sequences.

4. **Combined Dataset**: A combined dataset that merges data from the emoticon, feature, and text sequence datasets is created and dimensionality is reduced using PCA. A logistic regression model is trained and evaluated on this combined dataset.

### Models
1. **EmoticonModel**: Preprocesses emoticons, removes common emojis, tokenizes the input, and uses an LSTM model to extract features. These features are then classified using Logistic Regression.

2. **FeaturesModel**: Loads feature data, reshapes it, and uses Random Forest classifiers to predict labels. The model evaluates accuracy for various subsets of the training data.

3. **TextSeqModel**: Preprocesses text sequence data by removing specific substrings, encodes the text, and uses an LSTM model to classify the data. Performance is evaluated based on different percentages of training data.

4. **CombinedModel**: Combines the feature data from Emoticon, Features, and Text Sequence datasets, applies PCA for dimensionality reduction, and trains a logistic regression model on the combined dataset. The performance is evaluated and predictions are saved.

### Key Files
- `train_emoticon.csv`, `valid_emoticon.csv`, `test_emoticon.csv`: Datasets for emoticon-based classification.
- `train_feature.npz`, `valid_feature.npz`, `test_feature.npz`: Feature-based datasets for Random Forest classification.
- `train_text_seq.csv`, `valid_text_seq.csv`, `test_text_seq.csv`: Datasets for text sequence-based classification.
- `pred_emoticon.txt`, `pred_deepfeat.txt`, `pred_combined.txt`: Text files containing the predictions for emoticon, feature, and combined models respectively.

### How to Run
1. Ensure that you have the required libraries installed:
   numpy , pandas, sklearn , matplotlib
2. python3 ML_Models.py

## MiniProject2 - Image Classifier 

This project demonstrates the implementation of an **image multiclass classification model** using a **Progressive Model Adaptation (PMA)** approach. The model is built using a **pre-trained ResNet50** for feature extraction, combined with **Learn-with-Prototype (LwP)** and **Cosine Similarity** to perform efficient classification.

### Overview

In this project, we use **ResNet50**, a powerful pre-trained convolutional neural network, for feature extraction. The features are then utilized in a classification task to predict the class of each image in a multiclass setup. Key aspects of the approach include:

1. **Progressive Model Adaptation (PMA)**: This method allows the model to adapt progressively to new classes by learning class prototypes dynamically, helping the model generalize better and efficiently handle new data distributions.
  
2. **Learn-with-Prototype (LwP)**: In this approach, each class is represented by a **prototype vector**. The model learns to predict which class a given image belongs to by calculating the similarity between the image's feature vector and the class prototypes.

3. **Cosine Similarity**: Cosine similarity is used to calculate the similarity between an image’s feature vector and the class prototypes. The class whose prototype has the highest cosine similarity to the image feature is selected as the predicted class.

### Key Components

#### 1. **ResNet50 for Feature Extraction**
   - **ResNet50** is a state-of-the-art deep convolutional neural network pre-trained on **ImageNet**. It is designed to efficiently extract feature representations from input images by using residual learning, which helps in handling the vanishing gradient problem.
   - In this project, we use **ResNet50** **as a feature extractor**, where the final convolutional layers' output is used as the feature vector for each image. These extracted features are then used in the classification task.

#### 2. **Learn-with-Prototype (LwP)**
   - **Learn-with-Prototype (LwP)** allows the model to represent each class with a **prototype vector**. This prototype represents the center or average feature vector of the images in that class.
   - The model dynamically learns the class prototypes during training, progressively adapting to new data. 
   - During classification, the cosine similarity between the feature vector of the test image and the prototypes of all classes is computed. The class with the highest cosine similarity is selected as the predicted class.

#### 3. **Cosine Similarity**
   - **Cosine Similarity** is used to measure the similarity between the feature vector of an image and the prototypes of each class. The class whose prototype is most similar (based on cosine similarity) to the feature vector of the image is predicted as the class for that image.
   - **Cosine similarity** is used **once** during the classification process to determine the most similar class prototype to the image's feature vector.

### Workflow

1. **Data Preprocessing**:
   - Resize images to a consistent size suitable for **ResNet50** (e.g., 224x224 pixels).
   - Normalize the images to have pixel values in the range [0, 1].
   
2. **Feature Extraction**:
   - Use the pre-trained **ResNet50** model to extract feature vectors from images. These feature vectors represent the images in a high-dimensional feature space.
   
3. **Learn-with-Prototype (LwP)**:
   - For each class, calculate the prototype as the **mean feature vector** of all images belonging to that class. 
   - During training, update the prototypes progressively as the model adapts to new data.

4. **Classification**:
   - For each test image, calculate its feature vector using **ResNet50**.
   - Calculate the **cosine similarity** between the test image's feature vector and the prototypes of each class.
   - Assign the test image to the class with the highest cosine similarity.

5. **Model Training**:
   - Train the model using the training dataset, progressively updating the prototypes as new data is processed.
   - The training process involves comparing the extracted features of training images with the current prototypes and updating them accordingly.

6. **Prediction and Evaluation**:
   - After training, the model can be used to predict the classes of new images by comparing their feature vectors with the stored prototypes.
   - Model performance can be evaluated using classification metrics such as accuracy, precision, recall, and F1-score.

### Advantages of the Approach

1. **Efficient Learning with Limited Data**: By using **ResNet50** pre-trained on **ImageNet**, the model can generalize well with limited labeled data.
2. **Progressive Adaptation**: The model adapts progressively to new classes, making it scalable and able to learn from new data without forgetting previous classes.
3. **Robust to Similarity**: By using **cosine similarity**, the model effectively measures the closeness of feature vectors, ensuring that the most similar prototype is selected for classification.

## Team
1. Mohd Nasar Siddiqui
2. Kartik
3. Abhishek Kumar
4. Param Soni
5. Samrat Patil
