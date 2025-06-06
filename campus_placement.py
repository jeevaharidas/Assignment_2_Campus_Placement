import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path="train.csv"):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        print("Shape of the dataset:", df.shape)
        df.info()
        print("\nMissing values:")
        print(df.isnull().sum())
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def preprocess_data(df):
    if df is None:
        return None, None, None, None

    print("\n--- Starting Data Preprocessing ---")

    if 'sl_no' in df.columns:
        df = df.drop('sl_no', axis=1)
    if 'salary' in df.columns:
        df = df.drop('salary', axis=1)

    X = df.drop('status', axis=1)
    y_raw = df['status']

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, preprocessor, le

def evaluate_model(model_name, model, X_test_processed, y_test, status_labels):
    print(f"\n--- Evaluating: {model_name} ---")
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=status_labels))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=status_labels, yticklabels=status_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return accuracy, precision, recall, f1

def main():
    df = load_data()
    if df is None:
        return

    X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess_data(df)
    status_labels = label_encoder.classes_

    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest (Default)": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "k-Nearest Neighbors": KNeighborsClassifier()
    }

    results = {}

    pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        print(f"\n--- Training: {name} ---")
        pipeline.fit(X_train, y_train)
        acc, prec, rec, f1 = evaluate_model(name, pipeline, X_test, y_test, status_labels)
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}
        pipelines[name] = pipeline

    print("\n--- Hyperparameter Tuning for Random Forest ---")
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    rf_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")

    best_rf = grid_search.best_estimator_
    acc, prec, rec, f1 = evaluate_model("Random Forest (Tuned)", best_rf, X_test, y_test, status_labels)
    results["Random Forest (Tuned)"] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}

    print("\n--- Voting Classifier ---")
    voting_clf = Pipeline([
        ('preprocessor', preprocessor),
        ('voting_classifier', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(solver='liblinear', random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42)),
                ('rf', best_rf.named_steps['classifier'])
            ],
            voting='hard'
        ))
    ])
    voting_clf.fit(X_train, y_train)
    acc, prec, rec, f1 = evaluate_model("Voting Classifier", voting_clf, X_test, y_test, status_labels)
    results["Voting Classifier"] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}

    print("\n--- Overall Model Performance Summary ---")
    print(pd.DataFrame(results).T.sort_values(by='F1-score', ascending=False))

if __name__ == '__main__':
    main()
