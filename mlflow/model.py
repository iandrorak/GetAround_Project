### Imports

import mlflow
from mlflow.models.signature import infer_signature

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score


if __name__ == "__main__":
    # Set the MLflow tracking environment
   
    ### Load data
    ga_princing = pd.read_csv('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv')



    ### Remove rows with outliers
    # Remove the negative mileage
    ga_princing = ga_princing[ga_princing['mileage'] > 0]

    # Remove the rows with engine_power = 0
    ga_princing = ga_princing[ga_princing['engine_power'] > 0]



    ### Remove irrelevant columns
    # Remove paint_color column
    ga_princing = ga_princing.drop(columns=['paint_color'])

    # Remove Unnamed: 0 column and reset index
    ga_princing = ga_princing.drop(columns=['Unnamed: 0']).reset_index(drop=True)



    ### Preprocessing
    # Get numerical and categorical columns lists
    feature_list = ga_princing.columns.to_list()
    target_variable = feature_list.pop(feature_list.index('rental_price_per_day'))
    categorical_features = feature_list.copy()
    numerical_features = [categorical_features.pop(categorical_features.index('mileage'))]
    numerical_features.append(categorical_features.pop(categorical_features.index('engine_power')))

    # Separate target variable Y from features X
    X = ga_princing.loc[:, feature_list]
    Y = ga_princing.loc[:, target_variable]
    print('X = ',X.head())
    print('Y = ',Y.head())

    # Divide dataset Train set & Test set 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create a pipeline for the numerical features
    numrical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Create a pipeline for the categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])

    # Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numrical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Preprocess the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)



    ### Modeling
    ## Create a linear regression model
    lin_model = LinearRegression()
    lin_model.fit(X_train_preprocessed, Y_train)

    ## Create a ridge regression model
    # Perform grid search
    params_ridge = {'alpha': [0.1, 1, 10, 100, 1000]}
    grid_ridge = GridSearchCV(Ridge(), params_ridge, cv=3)
    grid_ridge.fit(X_train_preprocessed, Y_train)
    print('Best parameters for lin: ', grid_ridge.best_params_)


    ## Create a random forest regression model
    # Perform grid search
    params_rf = {'n_estimators': [10, 100, 1000],
                'max_depth': [2, 4, 6, 8, 10],
                'max_features': [1.0, 'sqrt', 'log2'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],}
    grid_rf = GridSearchCV(RandomForestRegressor(), params_rf, cv=3)
    grid_rf.fit(X_train_preprocessed, Y_train)
    print('Best parameters for rf: ', grid_rf.best_params_)

    ## Create a gradient boosting regression model
    # Perform grid search
    params_gbr = {'n_estimators': [10, 100, 1000],
                'max_depth': [2, 4, 6, 8, 10],
                'max_features': [1.0, 'sqrt', 'log2'],
                'subsample': [0.5, 0.75, 1],
                'learning_rate': [0.1, 0.5, 1]
                }
    grid_gbr = GridSearchCV(GradientBoostingRegressor(), params_gbr, cv=3)
    grid_gbr.fit(X_train_preprocessed, Y_train)
    print('Best parameters for gbr: ', grid_gbr.best_params_)

    ## Create an AdaBoost regression model
    # Perform grid search
    params_ada = {'n_estimators': [10, 100, 1000],
                'learning_rate': [0.1, 0.5, 1],
                'loss' : ['linear']
                }
    grid_ada = GridSearchCV(AdaBoostRegressor(), params_ada, cv=3)
    grid_ada.fit(X_train_preprocessed, Y_train)
    print('Best parameters for ada: ', grid_ada.best_params_)

    ## Create a voting regression model with a pipeline
    voting_regressor = Pipeline(steps=[
                                        ('preprocessor', preprocessor),
                                        ('vote', VotingRegressor(estimators=[('lin', lin_model),
                                                        ('ridge', grid_ridge.best_estimator_),
                                                        ('rf', grid_rf.best_estimator_),
                                                        ('gbr', grid_gbr.best_estimator_),
                                                        ('ada', grid_ada.best_estimator_)
                                                        ]))
                                    ]
    )




    ### Log Voting Regressor model to MLflow
    # Set your variables for your environment
    EXPERIMENT_NAME="voting-regressor-experiment"
    # Set experiment's info 
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)


    client = mlflow.tracking.MlflowClient()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    run = client.create_run(experiment.experiment_id) # Creates a new run for a given experiment

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_id=run.info.run_id) as run:
    
        # Fit the model
        voting_regressor.fit(X_train, Y_train)  # Fit the model
        predictions = voting_regressor.predict(X_train)  # Predict on the training set

        # Store metrics 
        
        # Log model seperately to have more flexibility on setup 
        mlflow.sklearn.log_model(
            sk_model= voting_regressor,
            artifact_path="pricing-model",
            registered_model_name="voting-regressor-model",
            signature=infer_signature(X_train, predictions) # Infer signature to tell what should be as model's inputs and outputs
        )

        # Print results 
        print("Voting Regressor Model")