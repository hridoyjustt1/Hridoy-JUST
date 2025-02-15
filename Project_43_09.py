import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset
df=pd.read_csv("E:\\Breast_Cancer.csv")

# Encode categorical columns
label_encoder = LabelEncoder()
df = df.apply(lambda col: label_encoder.fit_transform(col) if col.dtypes == 'object' else col)

# Define features and target
x = df.iloc[:, :-1].values  # All columns except the last as features
y = df.iloc[:, -1].values   # Last column as target

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize classifiers
svc_classifier = SVC(kernel='rbf', random_state=0, verbose=True)
dt_classifier = DecisionTreeClassifier(random_state=0)
k_classifier = KNeighborsClassifier(n_neighbors=3)

# Train and evaluate SVC
svc_classifier.fit(x_train, y_train)
svc_y_pred = svc_classifier.predict(x_test)
svc_acc = accuracy_score(y_test, svc_y_pred)
svc_f1 = f1_score(y_test, svc_y_pred)
svc_recall_sc = recall_score(y_test, svc_y_pred)
svc_pre_sc = precision_score(y_test, svc_y_pred)

# Train and evaluate Decision Tree
dt_classifier.fit(x_train, y_train)
dt_y_pred = dt_classifier.predict(x_test)
dt_acc = accuracy_score(y_test, dt_y_pred)
dt_f1 = f1_score(y_test, dt_y_pred)
dt_recall_sc = recall_score(y_test, dt_y_pred)
dt_pre_sc = precision_score(y_test, dt_y_pred)

# Train and evaluate KNN
k_classifier.fit(x_train, y_train)
k_y_pred = k_classifier.predict(x_test)
k_acc = accuracy_score(y_test, k_y_pred)
k_f1 = f1_score(y_test, k_y_pred)
k_recall_sc = recall_score(y_test, k_y_pred)
k_pre_sc = precision_score(y_test, k_y_pred)

# Print results
print("SVC:", [svc_acc, svc_f1, svc_recall_sc, svc_pre_sc])
print("Decision Tree:", [dt_acc, dt_f1, dt_recall_sc, dt_pre_sc])
print("KNN:", [k_acc, k_f1, k_recall_sc, k_pre_sc])

# Plot results
metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
data = np.array([[svc_acc, svc_f1, svc_recall_sc, svc_pre_sc],
                 [dt_acc, dt_f1, dt_recall_sc, dt_pre_sc],
                 [k_acc, k_f1, k_recall_sc, k_pre_sc]])

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(metrics))
width = 0.2

ax.bar(x_pos - width, data[0], width, label='SVC')
ax.bar(x_pos, data[1], width, label='Decision Tree')
ax.bar(x_pos + width, data[2], width, label='KNN')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Algorithms Performance')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()