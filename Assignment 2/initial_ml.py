import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Read Data
df = pd.read_csv(r"..\data\SAML-D.csv")
dfc = df.copy()

# Feature Engineering
dfc["Datetime"] = pd.to_datetime(dfc["Date"] + ' ' + dfc["Time"], format="%Y-%m-%d %H:%M:%S")
dfc["weekday"] = dfc["Datetime"].dt.day_name()
dfc["month"] = dfc["Datetime"].dt.month_name()
dfc["hour"] = dfc["Datetime"].dt.hour
dfc["sender_receiver_location"] = dfc["Sender_bank_location"] + "-" + dfc["Receiver_bank_location"]
dfc["sender_receiver_currency"] = dfc["Payment_currency"] + "-" + dfc["Received_currency"]

# Select X and y
y = dfc["Is_laundering"]
X = dfc[["Amount", "Payment_type", "weekday", "month", "hour", "sender_receiver_location", "sender_receiver_currency"]]

# Encode categorical X columns
cat_cols = ["Payment_type", "weekday", "month", "hour", "sender_receiver_location", "sender_receiver_currency"]
ct = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            cat_cols
        ),
        (
            "scaler",
            StandardScaler(),
            ["Amount"]
        )
    ],
    remainder="drop"
)

# Appproach 1: CLass Weights
print("Approach 1: Class Weights")

# Train-Test Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train = ct.fit_transform(X_train_raw)
X_test = ct.transform(X_test_raw)

# Check distribution of y in train set and test set
print("\nChecking y distribution:")

print("\nOverall y:")
print(y.value_counts(normalize=True))

print("\nTrain Set:")
print(y_train.value_counts(normalize=True))

print("\nTest Set:")
print(y_test.value_counts(normalize=True))

# Logistic Regression
logreg_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="saga")
logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, logreg_pred))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logreg_pred, normalize="true"))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logreg_pred))

joblib.dump(logreg_model, r"..\models\logreg.pkl")

# CART
cart_model = DecisionTreeClassifier(min_samples_split=50, max_depth=5, class_weight="balanced", random_state=42)
cart_model.fit(X_train, y_train)
cart_pred = cart_model.predict(X_test)

print("\nCART Accuracy:")
print(accuracy_score(y_test, cart_pred))

print("\nCART Confusion Matrix:")
print(confusion_matrix(y_test, cart_pred, normalize="true"))

print("\nCART Classification Report:")
print(classification_report(y_test, cart_pred))

joblib.dump(cart_model, r"..\models\cart.pkl")

# Random Forest
rf_model = RandomForestClassifier(min_samples_split=50, max_depth=5, n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred, normalize="true"))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

joblib.dump(rf_model, r"..\models\rf.pkl")


# Appproach 2: Undersampling
print("\n\nApproach 2: Undersampling")

us = RandomUnderSampler(random_state=42)
X_train_us, y_train_us = us.fit_resample(X_train, y_train)

# Check distribution of y in train set and test set
print("\nChecking y distribution:")

print("\nTrain Set:")
print(y_train_us.value_counts(normalize=True))

# Logistic Regression
logreg_model_us = LogisticRegression(max_iter=1000, random_state=42, solver="saga")
logreg_model_us.fit(X_train_us, y_train_us)
logreg_pred_us = logreg_model_us.predict(X_test)

print("\nLogistic Regression (Undersampled) Accuracy:")
print(accuracy_score(y_test, logreg_pred_us))

print("\nLogistic Regression (Undersampled) Confusion Matrix:")
print(confusion_matrix(y_test, logreg_pred_us, normalize="true"))

print("\nLogistic Regression (Undersampled) Classification Report:")
print(classification_report(y_test, logreg_pred_us))

joblib.dump(logreg_model_us, r"..\models\logreg_us.pkl")

# CART
cart_model_us = DecisionTreeClassifier(min_samples_split=50, max_depth=5, random_state=42)
cart_model_us.fit(X_train_us, y_train_us)
cart_pred_us = cart_model_us.predict(X_test)

print("\nCART (Undersampled) Accuracy:")
print(accuracy_score(y_test, cart_pred_us))

print("\nCART (Undersampled) Confusion Matrix:")
print(confusion_matrix(y_test, cart_pred_us, normalize="true"))

print("\nCART (Undersampled) Classification Report:")
print(classification_report(y_test, cart_pred_us))

joblib.dump(cart_model_us, r"..\models\cart_us.pkl")

# Random Forest
rf_model_us = RandomForestClassifier(min_samples_split=50, max_depth=5, n_estimators=500, random_state=42, n_jobs=-1)
rf_model_us.fit(X_train_us, y_train_us)
rf_pred_us = rf_model_us.predict(X_test)

print("\nRandom Forest (Undersampled) Accuracy:")
print(accuracy_score(y_test, rf_pred_us))

print("\nRandom Forest (Undersampled) Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred_us, normalize="true"))

print("\nRandom Forest (Undersampled) Classification Report:")
print(classification_report(y_test, rf_pred_us))

joblib.dump(rf_model_us, r"..\models\rf_us.pkl")