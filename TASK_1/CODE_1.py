import os
import pandas as pd
from sklearn.linear_model import LinearRegression

# Specify the folder where the files are located
data_folder = "C:/Users/Lenovo/OneDrive/Desktop/TASK_1/"

# Load the train.csv file
train_file = os.path.join(data_folder, 'train.csv')
train_df = pd.read_csv(train_file)

# Load the test (1).csv file
test_file = os.path.join(data_folder, 'test (1).csv')
test_df = pd.read_csv(test_file)

# Prepare the features for the training data
train_df['TotalBath'] = train_df['FullBath'] + train_df['HalfBath'] * 0.5
X_train = train_df[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
y_train = train_df['SalePrice']

# Prepare the features for the test data
test_df['TotalBath'] = test_df['FullBath'] + test_df['HalfBath'] * 0.5
X_test = test_df[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
test_predictions = model.predict(X_test)

# Create a DataFrame for the submission using the IDs from the test data
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})

# Save the submission file
output_file = os.path.join(data_folder, 'submission.csv')
submission.to_csv(output_file, index=False)

print(submission.head())
