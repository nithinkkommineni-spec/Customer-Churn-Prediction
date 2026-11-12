
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from pathlib import Path

# Find the folder containing this Python script
project_folder = Path(__file__).resolve().parent

# Locate the CSV dataset
csv_path = project_folder / "customer_churn_data.csv"

# Load the dataset
df = pd.read_csv(csv_path)

# 2. Explore dataset
print(df.head())
print(df.info())
print(df.describe(include="all"))
print(df.isna().sum())

# 3. Analyze customer churn
print("\nCustomer Churn Distribution:")
print(df["Churn"].value_counts())

print("\nCustomer Churn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# 4. Define independent and dependent variables
X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2529, stratify=y
)

# 6. Identify categorical columns
categorical_cols = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# 7. Create preprocessing and prediction pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_cols)
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=2529))
])

# 8. Train the model
model.fit(X_train, y_train)

# 9. Predict customer churn
y_pred = model.predict(X_test)

# 10. Evaluate model performance
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
