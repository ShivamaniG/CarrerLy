import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load the dataset
career = pd.read_csv(r'C:\Users\venki\Downloads\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main\dataset9000.data', header=None)

# Set column names
career.columns = ["Database Fundamentals", "Computer Architecture", "Distributed Computing Systems",
                  "Cyber Security", "Networking", "Development", "Programming Skills", "Project Management",
                  "Computer Forensics Fundamentals", "Technical Communication", "AI ML", "Software Engineering",
                  "Business Analysis", "Communication skills", "Data Science", "Troubleshooting skills",
                  "Graphics Designing", "Roles"]

# Drop NaN values
career.dropna(how='all', inplace=True)

# Separate features (X) and target variable (y)
X = np.array(career.iloc[:, 0:17])  # X is skills
y = np.array(career.iloc[:, 17])  # Y is Roles

# Encode the target variable (y) into numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=524)

# Build and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate and print accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy =', accuracy * 100)

# Save the trained model
pickle.dump(knn, open('careerlast.pkl', 'wb'))
print('Trained model saved as careerlast.pkl')
