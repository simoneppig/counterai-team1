import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
        )
    ],
    remainder="passthrough"
)

Xt = ct.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.1)

# Check distribution of y in train set and test set
print("Overall y:")
print(y.value_counts(normalize=True))

print("\nTrain Set:")
print(y_train.value_counts(normalize=True))

print("\nTest Set:")
print(y_test.value_counts(normalize=True))

# Logistic Regression
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, logreg_pred))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logreg_pred, normalize="true"))

# CART
cart_model = DecisionTreeClassifier(min_samples_split=20, max_depth=5)
cart_model.fit(X_train, y_train)
cart_pred = cart_model.predict(X_test)

print("\nCART Accuracy:")
print(accuracy_score(y_test, cart_pred))

print("\nCART Confusion Matrix:")
print(confusion_matrix(y_test, cart_pred, normalize="true"))

# Random Forest
rf_model = RandomForestClassifier(min_samples_split=20, max_depth=5, n_estimators=200)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred, normalize="true"))