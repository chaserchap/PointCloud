import psycopg2
from resources.creds import *
import pandas as pd
import xgboost as xgb

import sys

def xgb_predict(tablename):
    conn_info = {
        "host": "localhost",
        "port": 5432,
        "database": "pointclouds",
        "user": username,
        "password": password
    }

    conn = psycopg2.connect(**conn_info)
    curs = conn.cursor("named")

    sys.stdout.write(f"Connected to database 'pointclouds'. Extracting points from {tablename}.\n")

    cols = ['linearity', 'planarity', 'scattering', 'verticality']
    curs.execute(f"SELECT {','.join(cols)} FROM {tablename}")

    df = pd.DataFrame(curs.fetchall(), columns=cols)
    curs.close()

    df.astype(float)

    sys.stdout.write(f"Points retrieved. Loading xgb model.\n")

    model = joblib.load("models/xgboostv1.1.dat")

    sys.stdout.write(f"Model loaded, making predictions...\n")

    model.predict(df)


