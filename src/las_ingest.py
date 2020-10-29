from pdal import Pipeline
import json, sys


def ingest_data(las_file, tablename=None, db=None):
    if tablename is None:
        tablename = las_file.split('/')[-1].split('.')[0]

    sys.stdout.write(f"Adding {las_file} to table {tablename} in {db}.\n")

    pipe = {'pipeline': [
        {'type': 'readers.las',
         'filename': las_file},
        {"type": "filters.chipper",
         "capacity": 400},
        {"type": "writers.pgpointcloud",
         "connection": f"host='127.0.0.1' dbname={db} user={username} password={password}",
         "table": tablename,
         "overwrite": True}
    ]}

    pipeline = Pipeline(json.dumps(pipe))
    pipeline.execute()
    sys.stdout.write(f"{tablename} added successfully.\n")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Ingest las data into the database.")
    parser.add_argument('-f', '--filename', help='A lidar file to be ingested into the database, of filetype .las')
    parser.add_argument('-t', '--tablename', default=None, help='Name of the table in the database to put the data.')
    parser.add_argument('-d', '--database', default='pointclouds', help='Name of the database to store point cloud data.')
    args = parser.parse_args()

    ingest_data(
        las_file=args.filename,
        tablename=args.tablename,
        db=args.database
    )
