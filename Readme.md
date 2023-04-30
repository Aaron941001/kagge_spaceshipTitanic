Spaceship Titanic: Alternate Dimension Prediction

Achievements
This project achieved a high rank of 98/2590 with a score of 0.81038 on the Spaceship Titanic Kaggle competition.

![Kaggle Score](source/98%3A2590.png)


Introduction
This project targets the Kaggle competition "Spaceship Titanic: Alternate Dimension Prediction," which aims to predict whether passengers were transported to an alternate dimension during the collision of the spaceship using their personal data. We will accomplish this task by preprocessing the data, performing exploratory data analysis (EDA), and building a predictive model.

Model Design and Implementation
In the model.py script, we use two different machine learning models: XGBoost and CatBoost. First, we split the data into training and validation sets to evaluate the performance of the models.

Next, we use GridSearchCV and RandomizedSearchCV to tune the hyperparameters of the models. These two methods employ grid search and random search, respectively, to find the best combination of hyperparameters. After tuning, we retrain the models with the best hyperparameters and evaluate their performance on the validation set.

Development Environment and Dependencies
This project is developed using Python 3.8 and depends on the following libraries:

pandas 1.4.2
numpy 1.22.3
scikit-learn 1.1.2
matplotlib 3.5.2
seaborn 0.12.2
xgboost 1.7.5
catboost 1.1.1
Project Structure
.
├── preprocessed.py          # Data preprocessing script
├── EDA.py                   # Exploratory Data Analysis script
├── model.py                 # Model training and prediction script
├── train.csv                # Original training dataset
├── test.csv                 # Original test dataset
├── sample_submission.csv    # Example submission file
└── README.md                # Documentation
