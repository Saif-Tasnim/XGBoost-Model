import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import time

# Step 1: Read the original dataset
df = pd.read_csv('C:/Users/ABCD/Desktop/Computer Security/Datasets/transaction - used.csv')

# Step 2: Oversample the dataset to address class imbalance
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(df.drop('isFraud', axis=1), df['isFraud'])

# Step 3: Reduce the resampled dataset to 20k entries
df_sampled = pd.DataFrame(X_resampled, columns=df.drop('isFraud', axis=1).columns)
df_sampled['isFraud'] = y_resampled
df_sampled = df_sampled.sample(n=20000, random_state=42)

# Step 4: Plot a bar chart for the balanced 'isFraud' column
plt.figure(figsize=(8, 6))
df_sampled['isFraud'].value_counts().plot(kind='bar')
plt.title('Distribution of isFraud (balanced)')
plt.xlabel('isFraud')
plt.ylabel('Count')
plt.show()

# Step 5: Preprocess the data
X = df_sampled.drop('isFraud', axis=1)
y = df_sampled['isFraud']

# Convert categorical variables into numerical representations
X['type'] = pd.factorize(X['type'])[0]
X['nameOrig'] = pd.factorize(X['nameOrig'])[0]
X['nameDest'] = pd.factorize(X['nameDest'])[0]

# Step 6: Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Step 7: Initialize the XGBoost classifier
xgb_model = XGBClassifier(n_estimators=100, random_state=42)

# Step 8: Train the XGBoost model and measure the time taken
start_time = time.time()
xgb_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Step 9: Predict the target variable for the training, validation, and testing sets
y_train_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)
y_test_pred = xgb_model.predict(X_test)

# Step 10: Calculate and print the accuracy for training, validation, and testing sets
train_accuracy = (y_train_pred == y_train).mean()
val_accuracy = (y_val_pred == y_val).mean()
test_accuracy = (y_test_pred == y_test).mean()

print("Training set accuracy:", train_accuracy)
print("Validation set accuracy:", val_accuracy)
print("Testing set accuracy:", test_accuracy)

# Step 11: Plot a line chart of the accuracy over different sets
plt.figure(figsize=(8, 6))
plt.plot(['Training', 'Validation', 'Testing'], [train_accuracy, val_accuracy, test_accuracy], marker='o')
plt.title('Accuracy')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Step 12: Print the total time taken for training
print("Training time:", train_time)
