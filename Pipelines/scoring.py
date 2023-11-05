import pickle
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from Utils.db_builder import MySQLBuilder

'''
Build a scoring pipeline (scoring.py) where you will:
a. Read a new table that you will have to create via randomly sampling 10 rows from
dataset.csv (this is just for testing purposes).
b. Load the previously trained model and predict the above table.
c. Append the prediction results to this table (or a different one, with the appropriate
logic).
d. Report a final performance metric of your choice.
'''


class Scorer():
    def __init__(self, db_name, user, pwd, host, table_name, df_path) -> None:
        print("\n > Initialized Scoring Pipeline.")
        self.db_name = db_name
        self.user = user
        self.password = pwd
        self.host = host
        self.table_name = table_name
        self.df_path = df_path

    def orchestrator(self):
        print("\n > Initialized Scoring process.")
        sample_data = self.get_random_sample()
        model = self.load_model()
        
        sample_data = pd.DataFrame(sample_data)
        variables = sample_data.drop("charges", axis = 1)
        encoded = self.preprocessing(variables)
        scaled = self.scale_numerical_features(encoded)

        predictions = self.predict(model=model, X=scaled)
        sample_data["predicted_charges"] = predictions

        data_with_preds = sample_data
        self.create_scoring_table(data_with_preds)
        self.get_prediction_score(data_with_preds)
        self.test_select_from_table()

    def get_random_sample(self):
        print("\n > Creating random samples.")
        self.db_helper = MySQLBuilder(
            db_name = self.db_name, 
            table_name = self.table_name, 
            df_path = self.df_path, 
            host = self.host, 
            user = self.user,
            pwd = self.password
        )
        select_query = f'''
        SELECT *
        FROM {self.table_name}
        ORDER BY RAND(123)
        LIMIT 100
        '''
        results = self.db_helper.execute_sql_query(select_query)
        column_names = ["col1", "age", "sex", "bmi", "children", "smoker", "region", "charges"]
        df = pd.DataFrame(results, columns=column_names)
        df.set_index("col1", inplace=True)
        return df

    def load_model(self):
        print("\n > Loading BestModel from memory.")
        with open('best_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model
        
    def preprocessing(self, df):
        df_encoded = pd.get_dummies(df, columns=['sex', "smoker", "region", "children"])
        return df_encoded
    
    def scale_numerical_features(self, df):
        scaler = StandardScaler()
        return scaler.fit_transform(df)

    def predict(self, model, X):
        print("\n > Predicting charges...")
        predictions = model.predict(X)
        return predictions
    
    def create_scoring_table(self, data):
        # use the sample data on the model and store both the data and the predictions in a new scoring table
        print("\n > Creating Scoring table on MySQL.")
        create_query = '''
        CREATE TABLE IF NOT EXISTS scoring_table  (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    age INT, 
                    sex VARCHAR(255), 
                    bmi DOUBLE, 
                    children INT, 
                    smoker VARCHAR(255),
                    region VARCHAR(255),
                    charges DOUBLE,
                    predicted_charges DOUBLE
                )
        '''
        self.db_helper.execute_sql_query(create_query)
        self.db_helper.load_custom_data(data=data, custom_table_name="scoring_table")        

    def get_prediction_score(self, data):
        y_true = data["charges"]
        y_pred = data["predicted_charges"]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        print(f"\n ----> 🧠 Reported performance on sample data for RMSE: {round(rmse, 2)} 🧠 <----")

    def test_select_from_table(self):
        test_query = '''
        SELECT * from scoring_table
        '''
        results = self.db_helper.execute_sql_query(test_query)
        column_names = ["col1", "age", "sex", "bmi", "children", "smoker", "region", "charges", "predicted_charges"]
        df = pd.DataFrame(results[:10], columns=column_names)
        df.set_index("col1", inplace=True)
        print("\n First records from scoring table: \n ")
        print(df.head())