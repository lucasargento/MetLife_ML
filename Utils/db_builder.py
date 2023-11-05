import mysql.connector 
import pandas as pd

'''
Within your code, create an instance of MySQL or PostgreSQL database, 
where you'll create a table named training dataset and insert the
dataset.csv contents.
'''

class MySQLBuilder():
    def __init__(self, db_name, table_name, df_path, host, user, pwd) -> None:
        print("\n > MySQL helper initialized. Building Database instance.")

        self.db_name = db_name
        self.table_name = table_name
        self.df_path = df_path

        self.conn = mysql.connector.connect(
            host= host,
            user= user,
            password= pwd,
            database= db_name
        )

    def create_training_table(self):
        # TODO save queries as .sql files in a /Queries folder and read the sql logic from here.
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name}  (
            id INT AUTO_INCREMENT PRIMARY KEY,
            age INT, 
            sex VARCHAR(255), 
            bmi DOUBLE, 
            children INT, 
            smoker VARCHAR(255),
            region VARCHAR(255),
            charges DOUBLE
        )
        """
        cursor = self.conn.cursor()
        cursor.execute(create_table_query)
        self.conn.commit()
        cursor.close()

    def load_csv_data(self):
        train_data = self.df_path
        data = pd.read_csv(train_data)

        cursor = self.conn.cursor()

        for _, row in data.iterrows():
            # TODO save queries as .sql files in a /Queries folder and read the sql logic from here.
            insert_query = f"INSERT INTO {self.table_name} (age, sex, bmi, children, smoker, region, charges) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(insert_query, tuple(row))

        self.conn.commit()
        cursor.close()  

    def load_custom_data(self, data, custom_table_name):
        cursor = self.conn.cursor()

        for _, row in data.iterrows():
            # TODO save queries as .sql files in a /Queries folder and read the sql logic from here.
            insert_query = f"INSERT INTO {custom_table_name} (age, sex, bmi, children, smoker, region, charges, predicted_charges) VALUES (%s, %s, %s, %s, %s, %s, %s,%s)"
            cursor.execute(insert_query, tuple(row))

        self.conn.commit()
        cursor.close() 

    def print_rows(self):
        # Print first 10 rows for testing purposes
        cursor = self.conn.cursor()

        select_query = f'''
        SELECT * FROM {self.table_name}
        '''
        cursor.execute(select_query)
        results = cursor.fetchall()
        self.conn.commit()
        cursor.close()

        for result in results[:10]:
            print(result)
    
    def execute_sql_query(self, query_string):
        cursor = self.conn.cursor()

        cursor.execute(query_string)

        results = cursor.fetchall()
        self.conn.commit()
        cursor.close()

        return results