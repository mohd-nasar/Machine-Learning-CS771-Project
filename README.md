# Machine Learning Project
This repository contains the implementation of a machine learning project for the CS771 course at IIT Kanpur, under the guidance of Professor Piyush Rai (Aug 2024 - Dec 2024). The project focuses on classifying emoticons using various machine learning models.

## Emoticon Classifier

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

## Image Classifier 
