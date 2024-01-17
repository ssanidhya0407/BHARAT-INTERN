import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the Titanic dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Data Exploration and Preprocessing
selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[selected_features].copy()  # Create a copy to avoid SettingWithCopyWarning
y = train_df['Survived']

# Handle missing values
X['Age'].fillna(X['Age'].mean(), inplace=True)
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the grid search object
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Use the best model from the grid search
best_model = grid_search.best_estimator_

# Calculate and print accuracy on the test set using the best model
best_test_accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy with Best Model: {best_test_accuracy}")

# Calculate and print accuracy on the overall dataset
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}")

# Evaluate the model on the test set
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")



# Function to classify survival
def classify_survival():
    try:
        # Get input values
        pclass = int(pclass_entry.get())
        sex = sex_var.get()
        age = float(age_entry.get())
        sibsp = int(sibsp_entry.get())
        parch = int(parch_entry.get())
        fare = float(fare_entry.get())
        embarked = embarked_var.get()

        # Validate numeric inputs
        if not (0 <= age <= 150 and 0 <= fare <= 1000):
            raise ValueError("Invalid numeric value. Please enter valid values for Age and Fare.")

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=selected_features)
        input_data = pd.get_dummies(input_data, columns=['Sex', 'Embarked'], drop_first=True)

        # Ensure all required columns are present
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0

        # Reorder columns to match the training set
        input_data = input_data[X_train.columns]

        # Make prediction
        prediction = model.predict(input_data)

        # Display the result
        result_label.config(text=f"Prediction: {'Survived' if prediction[0] == 1 else 'Not Survived'}", foreground="#3498db")

    except ValueError as e:
        messagebox.showwarning("Input Error", str(e))
        print(f"Error: {e}")

# Function to clear input fields
def clear_input():
    pclass_entry.delete(0, tk.END)
    sex_combobox.set('Male')
    age_entry.delete(0, tk.END)
    sibsp_entry.delete(0, tk.END)
    parch_entry.delete(0, tk.END)
    fare_entry.delete(0, tk.END)
    embarked_combobox.set('C')
    result_label.config(text="")

# Create the main GUI window
root = tk.Tk()
root.title("Titanic Survival Prediction")



