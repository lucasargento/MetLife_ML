from Utils.db_builder import MySQLBuilder
from Pipelines.training import Trainer
from Pipelines.scoring import Scorer

# TODO save credentials as Json and add to .gitignore
db_name = "MetLifeChallenge"
user = "lucas"
password = "foo123"
host = "localhost"
table_name = "training_dataset"
df_path = "Dataset/dataset.csv"

def main():    
    builder = MySQLBuilder(
        db_name = db_name, 
        table_name = table_name, 
        df_path = df_path, 
        host = host, 
        user = user,
        pwd = password
    )

    builder.create_training_table()
    builder.load_csv_data()
    trainer = Trainer(db_name, user, password, host, table_name, df_path)
    scorer = Scorer(db_name, user, password, host, table_name, df_path)

    trainer.orchestrator()
    scorer.orchestrator()

if __name__ == "__main__":
    main()