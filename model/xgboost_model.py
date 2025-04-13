import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_xgboost_models(csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Drop unnecessary columns
    df = df.drop(columns=['histogram', 'byteentropy'])

    # Separate features (X) and targets (y)
    X = df.drop(columns=['is_malware', 'avClass'])
    y_binary = df['is_malware']
    y_multi = df['avClass']

    # Split the data into training and testing sets for binary classification
    X_train, X_test, y_binary_train, y_binary_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    # Reset indices to ensure alignment
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_binary_train.reset_index(drop=True, inplace=True)
    y_binary_test.reset_index(drop=True, inplace=True)

    # Train the binary classification model
    binary_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    binary_model.fit(X_train, y_binary_train)

    # Evaluate the binary classification model
    y_binary_pred = binary_model.predict(X_test)
    print("Binary Classification Report:")
    print(classification_report(y_binary_test, y_binary_pred))

    # Get prediction probabilities for binary classification
    y_binary_prob = binary_model.predict_proba(X_test)

    # Extract confidence percentages (probability of the predicted class)
    confidence_percentages = [
        max(prob) * 100  # Convert to percentage
        for prob in y_binary_prob
    ]

    # Add confidence percentages to the test set for inspection
    X_test_with_confidence = X_test.copy()
    X_test_with_confidence['predicted_class'] = y_binary_pred
    X_test_with_confidence['confidence_percentage'] = confidence_percentages

    # Display the first few rows of the test set with confidence percentages
    print("\nTest Set with Predictions and Confidence Percentages:")
    print(X_test_with_confidence[['predicted_class', 'confidence_percentage']].head())

    return binary_model
# Example usage
csv_path = "output_data.csv"
binary_model = train_xgboost_models(csv_path)
