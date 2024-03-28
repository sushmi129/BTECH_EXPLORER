import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier

# Read the CSV file
file_path = "final_data.csv"
df = pd.read_csv(file_path)

# Encoding categorical variables
label_encoders = {}
for column in ["Gender", "Category", "College", "Branch"]:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Splitting the dataset
X = df[["Rank", "Gender", "Category"]]
y = df[["College", "Branch"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Random Forest classifier
base_classifier = RandomForestClassifier()
clf = MultiOutputClassifier(base_classifier)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Dump label encoders and model
joblib.dump(label_encoders, "label_encoders.pkl", compress=True)
joblib.dump(clf, "model.pkl", compress=True)
