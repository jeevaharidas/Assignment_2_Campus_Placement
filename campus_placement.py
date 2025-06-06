import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path="train.csv"):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        print("Shape of the dataset:", df.shape)
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nInfo:")
        df.info()
        print("\nMissing values:")
        print(df.isnull().sum())
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def preprocess_data(df):
    """Preprocesses the data for modeling."""
    if df is None:
        return None, None, None, None

    print("\n--- Starting Data Preprocessing ---")

    # Drop sl_no and salary (salary is target-leaking for status prediction)
    if 'sl_no' in df.columns:
        df = df.drop('sl_no', axis=1)
        print("Dropped 'sl_no' column.")
    if 'salary' in df.columns:
        df = df.drop('salary', axis=1)
        print("Dropped 'salary' column (as it's an outcome of placement).")

    # Define target variable y, features X
    if 'status' not in df.columns:
        print("Error: Target column 'status' not found in the dataframe.")
        return None, None, None, None
        
    X = df.drop('status', axis=1)
    y_raw = df['status']

    # Encode target variable 'status'
    le = LabelEncoder()
    y = le.fit_transform(y_raw) # Placed:1, Not Placed:0 (depends on alphabetical order or first seen)
    print(f"\nTarget variable 'status' encoded. Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")


    # Identify categorical and numerical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    print(f"\nNumerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # drop='first' to avoid multicollinearity
    ])

    # Create column transformer to apply different transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough' # Keep any other columns (though ideally all are handled)
    )
    
    print("\nPreprocessor created with StandardScaler for numerical and OneHotEncoder for categorical features.")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nData split into training and testing sets (70/30 split, stratified).")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, le # Return label encoder for status interpretation

def evaluate_model(model_name, model, X_test_processed, y_test, status_labels):
    """Evaluates a model and prints metrics and confusion matrix."""
    print(f"\n--- Evaluating: {model_name} ---")
    y_pred = model.predict(X_test_processed)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=status_labels, zero_division=0))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plotting Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=status_labels, yticklabels=status_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return accuracy, precision, recall, f1

def main():
    # Load data
    df_train = load_data()
    if df_train is None:
        return

    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor, label_encoder_status = preprocess_data(df_train)
    if X_train is None:
        print("Data preprocessing failed. Exiting.")
        return

    status_labels = label_encoder_status.classes_ # Get ['Not Placed', 'Placed'] or similar

    # --- Define Models ---
    log_reg = LogisticRegression(solver='liblinear', random_state=42)
    dec_tree = DecisionTreeClassifier(random_state=42)
    rand_forest = RandomForestClassifier(random_state=42)

    # Create full pipelines with preprocessor and model
    pipeline_log_reg = Pipeline([('preprocessor', preprocessor), ('classifier', log_reg)])
    pipeline_dec_tree = Pipeline([('preprocessor', preprocessor), ('classifier', dec_tree)])
    pipeline_rand_forest = Pipeline([('preprocessor', preprocessor), ('classifier', rand_forest)])

    models_to_evaluate = {
        "Logistic Regression": pipeline_log_reg,
        "Decision Tree": pipeline_dec_tree,
        "Random Forest (Default)": pipeline_rand_forest
    }

    results = {}

    for name, model_pipeline in models_to_evaluate.items():
        print(f"\n--- Training: {name} ---")
        model_pipeline.fit(X_train, y_train)
        acc, prec, rec, f1 = evaluate_model(name, model_pipeline, X_test, y_test, status_labels)
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}
        
    # --- Hyperparameter Tuning for Random Forest (Example) ---
    print("\n--- Hyperparameter Tuning for Random Forest ---")

    # Define parameter grid for Random Forest

    param_grid_rf = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Note: 'classifier__' prefix is used because RandomForest is part of a pipeline
    grid_search_rf = GridSearchCV(estimator=pipeline_rand_forest, param_grid=param_grid_rf, 
                                  cv=3, n_jobs=-1, verbose=1, scoring='accuracy') # Using 3-fold CV for speed
    
    print("Starting GridSearchCV for Random Forest...")
    grid_search_rf.fit(X_train, y_train)
    
    print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")
    best_rand_forest = grid_search_rf.best_estimator_
    
    acc_rf_tuned, prec_rf_tuned, rec_rf_tuned, f1_rf_tuned = evaluate_model("Random Forest (Tuned)", best_rand_forest, X_test, y_test, status_labels)
    results["Random Forest (Tuned)"] = {'Accuracy': acc_rf_tuned, 'Precision': prec_rf_tuned, 'Recall': rec_rf_tuned, 'F1-score': f1_rf_tuned}


    # --- Voting Classifier ---
    print("\n--- Voting Classifier ---")
    # Setting the VotingClassifier estimators to the actual models, not the full pipeline, if applying preprocessing by itself or if the VotingClassifier itself sits inside some sort of larger pipeline.
    # To keep this example as straightforward as possible, let's say the base estimators get re-defined in pipelines once tuned; for default, the previously defined ones are used. 

    # Perhaps re-extract preprocessor for clarity or just use it.
    # Base estimators to the VotingClassifier in the pipeline should be ('name', model_object).
    # If the preprocessor is applied before the voting classifier:
    
    # Fit on the training data; transform on train/test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Base estimators for the Voting Classifier (using already trained/tuned models)
    # We need the final step of the pipeline (the classifier itself)
    clf1 = pipeline_log_reg.named_steps['classifier']
    clf2 = pipeline_dec_tree.named_steps['classifier']
    clf3_tuned = best_rand_forest.named_steps['classifier'] # Get the classifier from the tuned RF pipeline


    # Create Voting Classifier - it expects models that are already handling preprocessed data
    # or the voting classifier is put AFTER the preprocessor in its own pipeline.
    # Let's put it after the preprocessor.
    
    voting_clf_hard = VotingClassifier(
        estimators=[
            ('lr', clf1), 
            ('dt', clf2), 
            ('rf_tuned', clf3_tuned) 
        ], 
        voting='hard' # Hard voting: majority rule
    )

    # We need to train these base classifiers on the processed data if they are not already.
    # The pipelines above already trained them. So clf1, clf2, clf3_tuned are trained.
    # However, VotingClassifier internally fits clones of these estimators.
    # It's cleaner to define the VotingClassifier with untrained estimators and fit it on processed data.
    
    # Let's redefine base models (untrained) for the VotingClassifier
    log_reg_vc = LogisticRegression(solver='liblinear', random_state=42)
    dec_tree_vc = DecisionTreeClassifier(random_state=42)
    # Get best params for RF and create a new instance
    best_rf_params = {k.replace('classifier__', ''): v for k, v in grid_search_rf.best_params_.items()}
    rand_forest_vc_tuned = RandomForestClassifier(**best_rf_params, random_state=42)


    pipeline_voting_clf = Pipeline([
        ('preprocessor', preprocessor), # Apply preprocessing first
        ('voting_classifier', VotingClassifier(
            estimators=[
                ('lr', log_reg_vc), 
                ('dt', dec_tree_vc), 
                ('rf', rand_forest_vc_tuned) 
            ], 
            voting='hard'
        ))
    ])
    
    print("Training Voting Classifier (Hard Voting)...")
    pipeline_voting_clf.fit(X_train, y_train) # Fit on raw X_train, pipeline handles preprocessing
    
    acc_vote, prec_vote, rec_vote, f1_vote = evaluate_model("Voting Classifier (Hard)", pipeline_voting_clf, X_test, y_test, status_labels)
    results["Voting Classifier (Hard)"] = {'Accuracy': acc_vote, 'Precision': prec_vote, 'Recall': rec_vote, 'F1-score': f1_vote}

    # --- Summarize Results ---
    print("\n--- Overall Model Performance Summary ---")
    results_df = pd.DataFrame(results).T # Transpose to have models as rows
    print(results_df.sort_values(by='F1-score', ascending=False))

if __name__ == '__main__':
    main()