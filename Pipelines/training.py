import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from datetime import date
import pickle
from Utils.db_builder import MySQLBuilder
import traceback


'''
Build a training pipeline (training.py) where you will read this table, 
prepare it for modelling, run a hyperparameters search, store the best
trained model in any desired format, and store a text or pdf file with 
some evaluation metrics to describe the model.
'''


class Trainer():

    def __init__(self,db_name, user, pwd, host, table_name, df_path) -> None:
        print("\n > Initialized Training Pipeline.")

        self.db_name = db_name
        self.user = user
        self.password = pwd
        self.host = host
        self.table_name = table_name
        self.df_path = df_path

    def orchestrator(self):
        try:
            print("\n > Starting Training process.")
            
            dataset = self.read_training_table()
            df_encoded = self.preprocessing(dataset)
            X_train, X_test, y_train, y_test = self.train_test_split(df_encoded)
            X_train, X_test = self.scale_numerical_features(X_train = X_train, X_test = X_test)
            
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            self.train_models()
            self.evaluate_and_save_results()
            print("\n > Finished Training process succesfully")
        except Exception as e:
            print(f"\n > Exception during the training process: {e}")
            traceback.print_exc()

    def read_training_table(self):
        print("\n > Extracting data from MySQL database.")

        db_helper = MySQLBuilder(
            db_name = self.db_name, 
            table_name = self.table_name, 
            df_path = self.df_path, 
            host = self.host, 
            user = self.user,
            pwd = self.password
        )

        select_query = f'''
        SELECT * FROM {self.table_name}
        '''
        results = db_helper.execute_sql_query(select_query)
        column_names = ["col1", "age", "sex", "bmi", "children", "smoker", "region", "charges"]
        df = pd.DataFrame(results, columns=column_names)
        df.set_index("col1", inplace=True)
        print(f"\n > Showing first 5 rows: \n {df.head()}")
        return df

    def preprocessing(self, df):
        print("\n > Preprocessing data. Encoding categorical values with OneHotEncoding, deleting nulls if any, replacing missing values. etc.")
        df_encoded = pd.get_dummies(df, columns=['sex', "smoker", "region", "children"])
        return df_encoded

    def train_test_split(self, df):
        y = df.charges
        X = df.drop("charges", axis = 1, inplace = False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        splits = X_train, X_test, y_train, y_test
        print(f"\n > Performed an 80/20 split. Training set has {X_train.shape[0]} examples and {X_train.shape[1]} predictor variables. Test set has {X_test.shape[0]} examples.")
        return splits
    
    def scale_numerical_features(self,X_train, X_test):
        print("\n > Scaling numerical variables. Make sure this process is always performed after splitting your data to avoid data leakage.")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
    
    def train_models(self):
        print("\n > Training models. Performing HyperParameter tunning with Gridsearch cross validation. This could take a while..🧠")
        # TODO merge al metrics in a single dictionary
        self.maes = []
        self.r2 = []
        self.model_instances = []
        self.models = {"Linear Regression":0,
         "Polinomial Regression": 0,
          "Simple DT": 0,
          "Random Forest": 0,
          "Gradient Boosting":0,
          "MLP": 0,
        }

        self.train_linear_regression()
        self.train_polinomial_regression()
        self.train_dt_regressor()
        self.train_random_forest_regressor()
        self.train_gb_regressor()
        self.train_mlp_regressor()

    def train_linear_regression(self):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        model = LinearRegression()
        linear_grid_search = GridSearchCV(
            model,
            param_grid={},
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        linear_grid_search.fit(X_train, y_train)

        best_linear = linear_grid_search.best_estimator_
        self.model_instances.append(best_linear)

        y_pred_train = best_linear.predict(X_train)
        y_pred = best_linear.predict(X_test)

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse = mean_squared_error(y_test, y_pred)
        self.models["Linear Regression"] = mse
        self.maes.append(mean_absolute_error(y_test, y_pred))
        self.r2.append(r2_score(y_test, y_pred))

        print("\n > Results for Linear Regression\n")
        print(f"Mean Squared Error on the training set: {mse_train}")
        print(f"Mean Squared Error: {round(mse,2)}")
        print(f"Root Mean Squared Error: {round(np.sqrt(mse),2)}")
        print(f"MAE: {round(mean_absolute_error(y_test, y_pred),2)}")
        print(f"R2: {round(r2_score(y_test, y_pred),2)}")

    def train_polinomial_regression(self):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        polynomial_regressor = make_pipeline(PolynomialFeatures(), LinearRegression())
        poly_param_grid = {
            'polynomialfeatures__degree': [1, 2, 3, 4]
        }

        poly_grid_search = GridSearchCV(
            polynomial_regressor,
            param_grid=poly_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        poly_grid_search.fit(X_train, y_train)
        best_poly_model = poly_grid_search.best_estimator_
        self.model_instances.append(best_poly_model)

        y_pred_train = best_poly_model.predict(X_train)
        y_pred = best_poly_model.predict(X_test)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse = mean_squared_error(y_test, y_pred)
        self.models["Polinomial Regression"] = mse
        self.maes.append(mean_absolute_error(y_test, y_pred))
        self.r2.append(r2_score(y_test, y_pred))

        print("\n > Results for Polinomial Regression\n")
        print(f"Mean Squared Error on the training set: {mse_train}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {round(np.sqrt(mse),2)}")
        print(f"MAE: {round(mean_absolute_error(y_test, y_pred),2)}")
        print(f"R2: {round(r2_score(y_test, y_pred),2)}")

    def train_dt_regressor(self):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
 
        dt_param_grid = {
            'max_depth': [None, 3, 5, 7,10,20],
            'min_samples_split': [2, 5, 10]
        }
 
        dt_grid_search = GridSearchCV(
            DecisionTreeRegressor(),
            param_grid=dt_param_grid,
            cv=5,  
            scoring='neg_mean_squared_error', 
            n_jobs=-1 
        )
        
        dt_grid_search.fit(X_train, y_train)
        best_dt_regressor  = dt_grid_search.best_estimator_
        self.model_instances.append(best_dt_regressor)

        y_pred_train = best_dt_regressor.predict(X_train)
        y_pred = best_dt_regressor.predict(X_test)

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse = mean_squared_error(y_test, y_pred)
        self.models["Simple DT"] = mse
        self.maes.append(mean_absolute_error(y_test, y_pred))
        self.r2.append(r2_score(y_test, y_pred))
        
        print("\n > Results for Decision Tree Regressor\n")
        print(f"Mean Squared Error on the training set: {mse_train}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {round(np.sqrt(mse),2)}")
        print(f"MAE: {round(mean_absolute_error(y_test, y_pred),2)}")
        print(f"R2: {round(r2_score(y_test, y_pred),2)}")

    def train_random_forest_regressor(self):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }

        rf_grid_search = GridSearchCV(
            RandomForestRegressor(),
            param_grid=rf_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        rf_grid_search.fit(X_train, y_train)
        best_rf_regressor = rf_grid_search.best_estimator_
        self.model_instances.append(best_rf_regressor)

        y_pred_train = best_rf_regressor.predict(X_train)
        y_pred = best_rf_regressor.predict(X_test)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse = mean_squared_error(y_test, y_pred)
        self.models["Random Forest"] = mse
        self.maes.append(mean_absolute_error(y_test, y_pred))
        self.r2.append(r2_score(y_test, y_pred))

        print("\n > Results for Random Forest Regressor\n")
        print(f"Mean Squared Error on the training set: {mse_train}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {round(np.sqrt(mse),2)}")
        print(f"MAE: {round(mean_absolute_error(y_test, y_pred),2)}")
        print(f"R2: {round(r2_score(y_test, y_pred),2)}")

    def train_gb_regressor(self):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        gb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }

        gb_grid_search = GridSearchCV(
            GradientBoostingRegressor(),
            param_grid=gb_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        gb_grid_search.fit(X_train, y_train)
        best_gb = gb_grid_search.best_estimator_
        self.model_instances.append(best_gb)
        
        y_pred = best_gb.predict(X_test)
        y_pred_train = best_gb.predict(X_train)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse = mean_squared_error(y_test, y_pred)
        self.models["Gradient Boosting"] = mse
        self.maes.append(mean_absolute_error(y_test, y_pred))
        self.r2.append(r2_score(y_test, y_pred))

        print("\n > Results for Gradient Boosting Regressor\n")
        print(f"Mean Squared Error on the training set: {mse_train}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {round(np.sqrt(mse),2)}")
        print(f"MAE: {round(mean_absolute_error(y_test, y_pred),2)}")
        print(f"R2: {round(r2_score(y_test, y_pred),2)}")

    def train_mlp_regressor(self):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50)],
            'activation': ['relu'],
            'alpha': [0.01],
            'max_iter': [50000],
        }

        mlp_grid_search = GridSearchCV(
            MLPRegressor(),
            param_grid=param_grid,
            cv=5,  
            scoring='neg_mean_squared_error', 
            n_jobs=-1,
        )

        mlp_grid_search.fit(X_train, y_train)
        best_mlp = mlp_grid_search.best_estimator_
        self.model_instances.append(best_mlp)
        
        y_pred = best_mlp.predict(X_test)
        y_pred_train = best_mlp.predict(X_train)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse = mean_squared_error(y_test, y_pred)
        self.models["MLP"] = mse
        self.maes.append(mean_absolute_error(y_test, y_pred))
        self.r2.append(r2_score(y_test, y_pred))

        print("\n > Results for MLP Regressor\n")
        print(f"Mean Squared Error on the training set: {mse_train}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {round(np.sqrt(mse),2)}")
        print(f"MAE: {round(mean_absolute_error(y_test, y_pred),2)}")
        print(f"R2: {round(r2_score(y_test, y_pred),2)}")

    def evaluate_and_save_results(self):
        print(f"\n > Evaluating and saving best model.")

        mse_sq = []
        for x in self.models.values():
            mse_sq.append(np.sqrt(x))
        self.RMSES = mse_sq
        self.model_names = list(self.models.keys())
        self.MSES = list(self.models.values())
        
        
        # save results to txt
        filename = f'training_output - {date.today()}.csv'
        with open(filename, 'a') as file:
            file.write("model_name, MSE, RMSE, MAE, R2\n")
            for x in range(0, len(self.model_names)):
                line = str(self.model_names[x]) + "," + str(self.MSES[x])+","+ str(self.RMSES[x])+ "," + str(self.maes[x]) + "," + str(self.r2[x])
                file.write(line + "\n")
        print(f"Training results have been saved to {filename}")

        best_model = self.get_best_model()
        self.save_model(best_model)

    def get_best_model(self):
        # get best model based on the RMSE Metric
        index = self.RMSES.index(min(self.RMSES))
        best_model_name = self.model_names[index]
        print(f"\n ----> 🧠 Best model is {best_model_name} 🧠 <----")
        return self.model_instances[index]

    def save_model(self, best):
        # Dump model as pkl file
        with open('best_model.pkl', 'wb') as file:
            print(f"\n > Exporting best model to pkl file 'best_model.pkl'⬇️💾")
            pickle.dump(best, file)

