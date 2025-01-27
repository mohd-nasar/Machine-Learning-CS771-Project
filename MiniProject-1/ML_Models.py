import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load Data
def EmoticonModel():
    train_emoticon_df = pd.read_csv("train_emoticon.csv")
    valid_emoticon_df = pd.read_csv("valid_emoticon.csv")
    test_emoticon_df = pd.read_csv("test_emoticon.csv")

    # Define common emojis to remove
    common_emojis = "😑😛🛐😣🙯🚼🙼"

    # Function to remove common emojis
    def remove_common_emojis(text, emojis):
        return ''.join([char for char in text if char not in emojis])

    # Apply the function to remove emojis from input emoticons
    train_emoticon_df['cleaned_input'] = train_emoticon_df['input_emoticon'].apply(
        lambda x: remove_common_emojis(x, common_emojis))
    valid_emoticon_df['cleaned_input'] = valid_emoticon_df['input_emoticon'].apply(
        lambda x: remove_common_emojis(x, common_emojis))
    test_emoticon_df['cleaned_input'] = test_emoticon_df['input_emoticon'].apply(
        lambda x: remove_common_emojis(x, common_emojis))

    # Extract input emoticons and labels for training and validation
    train_emoticon_X = train_emoticon_df['cleaned_input'].tolist()
    train_emoticon_Y = train_emoticon_df['label'].tolist()

    valid_emoticon_X = valid_emoticon_df['cleaned_input'].tolist()
    valid_emoticon_Y = valid_emoticon_df['label'].tolist()

    test_emoticon_X = test_emoticon_df['cleaned_input'].tolist()

    # Tokenize emoticons (combine training, validation, and test data for fitting the tokenizer)
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(train_emoticon_X + valid_emoticon_X + test_emoticon_X)

    # Convert emoticons to sequences of token indices
    train_sequences = tokenizer.texts_to_sequences(train_emoticon_X)
    valid_sequences = tokenizer.texts_to_sequences(valid_emoticon_X)
    test_sequences = tokenizer.texts_to_sequences(test_emoticon_X)

    # Pad sequences to ensure they are of equal length
    max_length = max(len(seq) for seq in train_sequences + valid_sequences + test_sequences)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
    valid_padded = pad_sequences(valid_sequences, maxlen=max_length, padding='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    # Convert labels to integer encoding using LabelEncoder
    label_encoder = LabelEncoder()
    train_emoticon_Y = label_encoder.fit_transform(train_emoticon_Y)
    valid_emoticon_Y = label_encoder.transform(valid_emoticon_Y)

    # Define the RNN model
    vocab_size = len(tokenizer.word_index) + 1  # Plus 1 for padding token

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
        LSTM(2, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print the number of trainable parameters
    model.summary()

    # Split percentages for training
    train_splits = [0.2, 0.4, 0.6, 0.8, 1.0]
    validation_accuracies = []

    # Iterate through different training splits
    for train_split in train_splits:
        # Determine the number of samples for the current split
        split_idx = int(train_padded.shape[0] * train_split)
        train_padded_split = train_padded[:split_idx]
        train_emoticon_Y_split = train_emoticon_Y[:split_idx]

        # Train the LSTM model on the split data
        model.fit(train_padded_split, train_emoticon_Y_split, epochs=50, batch_size=32, verbose=0)

        # Extract features from the LSTM layer for training and validation data
        def extract_features(model, data):
            # Create a model that outputs the LSTM layer's output
            feature_extractor = Sequential(model.layers[:-1])  # Remove the last Dense layer
            return feature_extractor.predict(data)

        # Extract LSTM embeddings as features
        train_features = extract_features(model, train_padded_split)
        valid_features = extract_features(model, valid_padded)

        # Normalize the features for better performance with logistic regression
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        valid_features_scaled = scaler.transform(valid_features)

        # Train a logistic regression classifier on the LSTM features
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(train_features_scaled, train_emoticon_Y_split)

        # Evaluate logistic regression on the validation data
        valid_predictions = logreg.predict(valid_features_scaled)
        logreg_accuracy = accuracy_score(valid_emoticon_Y, valid_predictions)

        validation_accuracies.append(logreg_accuracy)
        print(f"Training Split {train_split * 100:.0f}%, Validation Accuracy using Logistic Regression on LSTM features: {logreg_accuracy:.4f}")

    # Plot training percentage vs validation accuracy
    plt.plot([v * 100 for v in train_splits], validation_accuracies, marker='o')
    plt.title('Training Percentage vs Validation Accuracy')
    plt.xlabel('Training Percentage (%)')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    # Predict on the test data using the logistic regression model
    test_features = extract_features(model, test_padded)
    test_features_scaled = scaler.transform(test_features)
    test_predictions = logreg.predict(test_features_scaled)

    # Print the predictions for the test data
    # Assuming test_predictions is already defined as a numpy array or a similar structure
    pred_df = pd.DataFrame(test_predictions)  # Create DataFrame without specifying column names
    pred_df.to_csv('pred_emoticon.txt', header=False, index=False)  # Save to .txt without header
    print("Prediciton file created for emoticon")


def FeaturesModel():

    # Load the datasets
    train_data = np.load('train_feature.npz')
    valid_data = np.load('valid_feature.npz')
    test_data = np.load('test_feature.npz')

    # Extract features and labels from the datasets
    X_train_full = train_data['features']
    y_train_full = train_data['label']
    X_valid = valid_data['features']
    y_valid = valid_data['label']
    X_test = test_data['features']

    # Ensure features are 2D arrays by flattening them if necessary
    X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Print shapes to verify correctness
    print(f"Training features shape: {X_train_full.shape}")
    print(f"Training labels shape: {y_train_full.shape}")
    print(f"Validation features shape: {X_valid.shape}")
    print(f"Validation labels shape: {y_valid.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Define the percentages of training data to be used
    percentages = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Initialize lists to store accuracies and number of parameters
    rf_train_accuracies = []
    rf_valid_accuracies = []
    rf_num_parameters_list = []

    # Function to count the number of parameters in a RandomForest model
    def count_rf_parameters(model):
        return sum(tree.tree_.node_count for tree in model.estimators_)

    # Define RandomForest hyperparameters
    rf_params = {
        'n_estimators': 10,      # Number of trees in the forest
        'max_depth': 10,         # Maximum depth of each tree
        'min_samples_split': 10, # Minimum samples to split an internal node
        'min_samples_leaf': 5    # Minimum samples required to be a leaf node
    }

    # Train the RandomForestClassifier with varying data sizes
    for pct in percentages:
        size = int(pct * len(X_train_full))
        X_train_subset = X_train_full[:size]
        y_train_subset = y_train_full[:size]

        # Train the RandomForest model
        rf_model = RandomForestClassifier(**rf_params, random_state=42)
        rf_model.fit(X_train_subset, y_train_subset)

        # Calculate training accuracy
        y_rf_train_pred = rf_model.predict(X_train_subset)
        train_accuracy_rf = accuracy_score(y_train_subset, y_rf_train_pred)
        rf_train_accuracies.append(train_accuracy_rf)

        # Calculate validation accuracy
        y_rf_valid_pred = rf_model.predict(X_valid)
        valid_accuracy_rf = accuracy_score(y_valid, y_rf_valid_pred)
        rf_valid_accuracies.append(valid_accuracy_rf)

        # Count the number of parameters in the RandomForest model
        rf_num_parameters = count_rf_parameters(rf_model)
        rf_num_parameters_list.append(rf_num_parameters)

        # Generate predictions on the test data
        test_predictions = rf_model.predict(X_test)

        # Save the predictions to a CSV file
        # Save the predictions to a text file without any labels
    test_predictions_df = pd.DataFrame(test_predictions)  # Create a DataFrame
    test_predictions_df.to_csv('pred_deepfeat.txt', header=False, index=False)  # Save to .txt without header

    print("Test predictions saved to 'pred_deepfeat.txt'.")



    # Plot: Training vs Validation Accuracy for RandomForest
    plt.subplot(2, 2, 2)
    plt.plot([int(p * 100) for p in percentages], rf_train_accuracies, marker='o', label='Training Accuracy')
    plt.plot([int(p * 100) for p in percentages], rf_valid_accuracies, marker='s', label='Validation Accuracy')
    plt.title('RandomForest: Training vs Validation Accuracy')
    plt.xlabel('Percentage of Training Data Used (%)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Annotate the number of parameters on the plot
    for i, txt in enumerate(rf_num_parameters_list):
        plt.annotate(f'Params: {txt}', (int(percentages[i] * 100), rf_valid_accuracies[i]))

    plt.tight_layout()
    plt.show()
def TextSeqModel():
    # Load and preprocess the datasets
    train_df = pd.read_csv("train_text_seq.csv")
    valid_df = pd.read_csv("valid_text_seq.csv")
    test_df = pd.read_csv("test_text_seq.csv")  # Load the test dataset

    substrings_to_remove = ['15436', '1596', '262', '415', '614', '422', '464', '284']


    # Function to remove the specified substrings
    def remove_substrings(sequence, substrings):
        for substring in substrings:
            sequence = sequence.replace(substring, '')  # Remove each substring
        return sequence


    # Preprocess the data
    def preprocess_data(df):
        df['processed_str'] = df['input_str'].apply(lambda x: x[3:])  # Remove leading zeros
        df['modified_str'] = df['processed_str'].apply(lambda x: remove_substrings(x, substrings_to_remove))
        df['modified_length'] = df['modified_str'].apply(len)
        df = df[df['modified_length'] == 13]  # Keep only strings with length 13 after modification
        return df


    # Preprocess train and validation data
    train_df = preprocess_data(train_df)
    valid_df = preprocess_data(valid_df)
    test_df = preprocess_data(test_df)  # Preprocess the test data


    # Convert the characters in modified_str to integers (character encoding)
    def encode_strings(df):
        all_chars = sorted(list(set("".join(df['modified_str'].values))))  # Get unique characters
        char_to_int = {char: i + 1 for i, char in enumerate(all_chars)}  # Map each character to an integer
        df['encoded_str'] = df['modified_str'].apply(lambda x: [char_to_int[char] for char in x])  # Encode each string
        return df, char_to_int


    train_df, char_to_int = encode_strings(train_df)
    valid_df, _ = encode_strings(valid_df)
    test_df, _ = encode_strings(test_df)  # Encode test data

    # Pad sequences to ensure all inputs have the same length (13, as we ensured)
    y_train = train_df['label'].values
    y_valid = valid_df['label'].values

    # Define training splits
    train_splits = [0.2, 0.4, 0.6, 0.8, 1.0]
    validation_accuracies = []

    # Iterate through different training splits
    for train_split in train_splits:
        # Select the portion of the training data
        split_idx = int(len(train_df) * train_split)
        X_train = pad_sequences(train_df['encoded_str'][:split_idx], maxlen=13, padding='post')

        # Define the LSTM model
        model = Sequential()
        model.add(Embedding(input_dim=len(char_to_int) + 1, output_dim=32, input_length=13))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train[:split_idx], epochs=50, batch_size=32, verbose=0,
                validation_data=(pad_sequences(valid_df['encoded_str'], maxlen=13, padding='post'), y_valid))

        # Evaluate the model on validation data
        y_pred = model.predict(pad_sequences(valid_df['encoded_str'], maxlen=13, padding='post'))
        y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

        # Calculate accuracy
        accuracy = accuracy_score(y_valid, y_pred)
        validation_accuracies.append(accuracy)
        print(f"Training Split {train_split * 100:.0f}%, Validation Accuracy: {accuracy * 100:.2f}%")

    # Predict on the test data
    X_test = pad_sequences(test_df['encoded_str'], maxlen=13, padding='post')
    y_test_pred = model.predict(X_test)
    y_test_pred = (y_test_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Save the predictions for the test data to a CSV file
     # Assuming y_test_pred is already defined and is a numpy array
    pred_df = pd.DataFrame(y_test_pred.flatten())  # Create a DataFrame without specifying column names
    pred_df.to_csv("pred_deepfeat.txt", header=False, index=False)  # Save to .txt without header



def CombinedModel():

    # Preprocessing function for text sequence dataset
    def preprocess_text_seq(df):
        X_text = df.iloc[:, :-1].values
        y_text = df['label'].values
        X_text_processed = np.array([list(map(int, list(seq))) for seq in X_text[:, 0]])
        return X_text_processed, y_text

    # Adjusted Preprocessing function for emoticon dataset
    def preprocess_emoticons(df, encoder=None):
        X_emoticons = df.iloc[:, :-1].values
        y_emoticons = df['label'].values
        X_emoticons_flat = np.array([list(seq) for seq in X_emoticons.flatten()])
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False)
            X_emoticons_encoded = encoder.fit_transform(X_emoticons_flat)
        else:
            X_emoticons_encoded = encoder.transform(X_emoticons_flat)
        return X_emoticons_encoded, y_emoticons, encoder

    # Preprocessing function for npz dataset
    def preprocess_npz(X_npz, y_npz):
        X_npz_flat = X_npz.reshape(X_npz.shape[0], -1)
        return X_npz_flat, y_npz

    # Combine all datasets function
    def combine_datasets(X_text, X_emoticons, X_npz):
        assert np.array_equal(y_text, y_emoticons) and np.array_equal(y_text, y_npz), "Labels don't match!"
        X_combined = np.concatenate([X_text, X_emoticons, X_npz], axis=1)
        return X_combined, y_text

    # Load the datasets
    text_seq_df = pd.read_csv('train_text_seq.csv')
    emoticon_df = pd.read_csv('train_emoticon.csv')
    train_npz = np.load('train_feature.npz')

    X_train_npz = train_npz['features']
    y_train_npz = train_npz['label']

    # Preprocess the training datasets
    X_text, y_text = preprocess_text_seq(text_seq_df)
    X_emoticons, y_emoticons, encoder = preprocess_emoticons(emoticon_df)
    X_npz_flat, y_npz = preprocess_npz(X_train_npz, y_train_npz)

    # Combine the training datasets
    X_combined, y_combined = combine_datasets(X_text, X_emoticons, X_npz_flat)

    # Scale the combined dataset
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)

    # Apply PCA to reduce the dimensions and maintain variance
    pca = PCA(n_components=0.95, random_state=42)  # Retain 95% variance
    X_combined_pca = pca.fit_transform(X_combined_scaled)

    # Train and evaluate the model on different percentages of combined data
    percentages = [0.2, 0.4, 0.6, 0.8, 1.0]
    accuracies = []

    for pct in percentages:
        size = int(pct * len(X_combined_pca))
        X_train_subset = X_combined_pca[:size]
        y_train_subset = y_combined[:size]

        # Train the logistic regression model
        model = LogisticRegression(C=0.5, penalty='l1', solver='liblinear', max_iter=1000)
        model.fit(X_train_subset, y_train_subset)

        # Evaluate on the complete combined validation data
        y_val_pred = model.predict(X_val_combined_pca)
        accuracy_val = accuracy_score(y_val_npz, y_val_pred)
        accuracies.append(accuracy_val)
        print(f"Validation Accuracy with {int(pct * 100)}% of training data: {accuracy_val:.4f}")

    # Plot the accuracies
    plt.figure(figsize=(8, 5))
    plt.plot([int(pct * 100) for pct in percentages], accuracies, marker='o', linestyle='-', color='b')
    plt.title('Validation Accuracy vs. Percentage of Training Data')
    plt.xlabel('Percentage of Training Data (%)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.show()

    # Load the test dataset
    test_npz = np.load('test_feature.npz')
    X_test_npz = test_npz['features']
    y_test_npz = test_npz['features']

    test_text_seq_df = pd.read_csv('test_text_seq.csv')
    test_emoticon_df = pd.read_csv('test_emoticon.csv')

    # Preprocess the test datasets
    X_test_text, y_test_text = preprocess_text_seq(test_text_seq_df)
    X_test_emoticons, y_test_emoticons, _ = preprocess_emoticons(test_emoticon_df, encoder)
    X_test_npz_flat, y_test_npz = preprocess_npz(X_test_npz, y_test_npz)

    # Combine the test datasets
    X_test_combined, y_test_ignored  = combine_datasets(X_test_text, X_test_emoticons, X_test_npz_flat)

    # Scale and apply PCA to the test data
    X_test_combined_scaled = scaler.transform(X_test_combined)
    X_test_combined_pca = pca.transform(X_test_combined_scaled)

    # Make predictions on the combined test data
    y_test_pred = model.predict(X_test_combined_pca)

    # Save the predictions to a CSV file
    # Save the predictions to a text file without any labels
    test_predictions_df = pd.DataFrame(y_test_pred)  # Create a DataFrame without specifying columns
    test_predictions_df.to_csv('pred_combined.txt', header=False, index=False)  # Save to .txt without header

    print("Test predictions saved to 'pred_combined.txt'.")


EmoticonModel()
FeaturesModel()
TextSeqModel()
CombinedModel()
