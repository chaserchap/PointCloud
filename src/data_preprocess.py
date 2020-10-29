from src.resources.config import *
from src.resources.resources import test_pipe
import pdal
import json
from src.resources import dbutils
import sys
from copy import deepcopy
import pandas as pd

def run_pipe(from_table, to_table):
    db_params = {
        "host": "localhost",
        "database": "pointclouds",
        "port": 5432,
        "user": username,
        "password": password
    }

    conn = dbutils.connect(db_params)
    sys.stdout.write("Connected to database 'pointclouds'.\n")
    this_pipe = deepcopy(test_pipe)
    reader = {
        "type": "readers.pgpointcloud",
        "connection": f"host=127.0.0.1 dbname='pointclouds' user={username} password={password}",
        "table": from_table,
        "column": "pa"
    }

    this_pipe["pipeline"].insert(0, reader)
    pipe = pdal.Pipeline(json.dumps(this_pipe))

    sys.stdout.write("Executing pdal pipeline.\n")
    pipe.execute()

    sys.stdout.write("pdal pipeline completed successfully.\n")
    dbutils.execute_mogrify(conn, pd.DataFrame(pipe.arrays[0]), to_table)
    sys.stdout.write(f"Added {from_table} to {to_table}.\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--from_table", help="Table with chipped point cloud (likely ingested using pdal.")
    parser.add_argument("-o", "--to_table", help="Table to output the results of the pdal pipeline.")

    args = parser.parse_args()

    run_pipe(args.from_table, args.to_table)