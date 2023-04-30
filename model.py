import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SequentialFeatureSelector


def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    return train_df, test_df

def prepare_data(train_df):
    X = train_df.drop(["PassengerId", "Transported"], axis=1) 
    y = train_df["Transported"].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    return X_train, X_val, y_train, y_val

def apply_SMOTE(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, y_train

def xgboost_hyperparameters():
    xgb_param_dist = {
        'subsample': [0.9], 'reg_lambda': [0.5], 'reg_alpha': [2], 'n_estimators': [500], 'max_depth': [5], 'learning_rate': [0.1], 'gamma': [1.5], 'colsample_bytree': [0.7]
    }
    return xgb_param_dist

def catboost_hyperparameters():
    cb_param_dist = {
        'learning_rate': [0.1], 'l2_leaf_reg': [1], 'iterations': [200], 'depth': [6], 'border_count': [64], 'bagging_temperature': [1]
    }
    return cb_param_dist

def randomized_search(estimator, param_dist, X_train, y_train):
    random_search = RandomizedSearchCV(estimator, param_distributions=param_dist, 
                                       n_iter=100, cv=5, verbose=2, n_jobs=-1, random_state=42)

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def compute_cross_val_score(estimator, X_train, y_train):
    return cross_val_score(estimator, X_train, y_train, cv=5).mean()

def plot_roc_curve(y_val, y_scores):
    fpr, tpr, _ = roc_curve(y_val, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def plot_cm(estimator, X_val, y_val):
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_confusion_matrix(estimator, X_val, y_val, ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.show()

def create_submission(test_df, best_clf):
    test_preds = best_clf.predict(test_df.drop(["PassengerId"], axis=1))
    test_preds_bool = test_preds.astype(bool)
    submission_df = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": test_preds_bool})
    submission_df.to_csv("submission.csv", index=False)

def fit_selector(selector, X_train, y_train):
    selector.fit(X_train, y_train)
    return selector

def transform_datasets(selector, X_train, X_val, test_df):
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(test_df.drop(["PassengerId"], axis=1))
    return X_train_selected, X_val_selected, X_test_selected

def print_confusion_matrix(y_val, val_preds):
    cm = confusion_matrix(y_val, val_preds)
    print("Confusion Matrix:")
    print(cm)

def fit_selector(selector, X_train, y_train):
    selector.fit(X_train, y_train)
    return selector

def transform_datasets(selector, X_train, X_val, test_df):
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(test_df.drop(["PassengerId"], axis=1))
    return X_train_selected, X_val_selected, X_test_selected


if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = prepare_data(train_df)
    X_train, y_train = apply_SMOTE(X_train, y_train)

    xgb_param_dist = xgboost_hyperparameters()
    xgb_clf = randomized_search(XGBClassifier(random_state=42), xgb_param_dist, X_train, y_train)

    cb_param_dist = catboost_hyperparameters()
    cb_clf = randomized_search(CatBoostClassifier(random_state=42, verbose=0, cat_features=None,
    loss_function='Logloss', eval_metric='Accuracy'), cb_param_dist, X_train, y_train)

    xgb_acc = compute_cross_val_score(xgb_clf, X_train, y_train)
    cb_acc = compute_cross_val_score(cb_clf, X_train, y_train)

    print("XGBoost Accuracy:", xgb_acc)
    print("CatBoost Accuracy:", cb_acc)

    best_clf = max((xgb_clf, xgb_acc), (cb_clf, cb_acc), key=lambda x: x[1])[0]

    selector = SequentialFeatureSelector(best_clf, n_features_to_select=5, direction='forward', scoring='accuracy', cv=5)
    selector = fit_selector(selector, X_train, y_train)

    X_train_selected, X_val_selected, X_test_selected = transform_datasets(selector, X_train, X_val, test_df)
    
    best_clf.fit(X_train_selected, y_train)

    best_clf.fit(X_train, y_train)
    val_preds = best_clf.predict(X_val)

    print_confusion_matrix(y_val, val_preds)

    y_scores = best_clf.predict_proba(X_val)[:, 1]
    plot_roc_curve(y_val, y_scores)
    plot_cm(best_clf, X_val, y_val)

    create_submission(test_df, best_clf)