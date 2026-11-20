
import pandas as pd
from sqlalchemy import create_engine, URL, text
from getpass import getpass

print("========== POSTGRESQL DATA UPLOAD ==========")

# Database credentials
password = getpass("Enter PostgreSQL password: ")

# Create database connection
connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password=password,
    host="localhost",
    port=5432,
    database="fraud_detection_db"
)

engine = create_engine(connection_url)

try:
    # Test connection
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    print("\nPostgreSQL connected successfully!")

    # Load prediction data
    df = pd.read_csv("outputs/transaction_predictions.csv")

    print("\nDataset loaded:", df.shape)
    print("Columns:", df.columns.tolist())

    # Upload predictions to PostgreSQL
    df.to_sql(
        name="transaction_predictions",
        con=engine,
        if_exists="fail",
        index=False,
        chunksize=5000
    )

    print("\nData uploaded successfully!")

    # Verify uploaded records
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT COUNT(*) FROM transaction_predictions")
        )
        count = result.scalar()

    print("Total records in PostgreSQL:", count)

except Exception as error:
    print("\nDatabase upload failed:", error)

finally:
    engine.dispose()
