import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Step 1: Feature Engineering
# Example data: This is a small set of example SQL query features and their corresponding memory usage in MB
data = {
    'num_joins': [0, 1, 2, 1, 3],
    'num_tables': [1, 2, 3, 2, 4],
    'big_tables': [0, 1, 1, 1, 2],
    #'tables': ['TableA', 'TableA TableB', 'TableA TableB thingdata', 'TableA TableB', 'TableA TableB thingdata TableC'],
    'ram_usage': [500, 1024, 5120, 1024, 10240]  # Memory consumption in MB (target)
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# Step 3: Preprocessing
# One-hot encode table names (categorical feature)
encoder = OneHotEncoder(sparse=False)
#table_encoded = encoder.fit_transform(df[['tables']])

# Concatenate the encoded tables with the other numerical features
#X = pd.concat([df[['num_joins', 'num_tables']], pd.DataFrame(table_encoded)], axis=1)
X = df[['num_joins', 'num_tables', 'big_tables']]
y = df['ram_usage']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection and Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 7: Use the trained model to make predictions on new SQL queries
new_query_features = {
    'num_joins': 3,
    'num_tables': 4,
    'big_tables': 1
    #'tables': 'TableA TableB thingdata'
}

# Preprocess the new query
new_query_df = pd.DataFrame([new_query_features])
#table_encoded_new_query = encoder.transform(new_query_df[['tables']])
#X_new_query = pd.concat([new_query_df[['num_joins', 'num_tables']], pd.DataFrame(table_encoded_new_query)], axis=1)
X_new_query = new_query_df[['num_joins', 'num_tables', 'big_tables']]


# Predict the memory (RAM) requirement for the new query
predicted_ram_usage = model.predict(X_new_query)
print(f"Predicted RAM usage: {predicted_ram_usage[0]} MB")
