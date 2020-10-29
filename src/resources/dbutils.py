import psycopg2
import numpy as np

translate_dtype = {'float64': 'numeric', 'float32': 'numeric', 'int64': 'integer', 'uint8': 'integer', 'uint16':'integer'}

def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1)
    print("Connection successful")
    return conn

def execute_mogrify(conn, df, table):
    """
    Using cursor.mogrify() to build the bulk insert query
    then cursor.execute() to execute the query
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    cursor = conn.cursor()
    num_vals = '(' + ','.join(['%s']*len(df.columns)) + ')'
    values = [cursor.mogrify(num_vals, tup).decode('utf8') for tup in tuples]
    query = "INSERT INTO %s(%s) VALUES " % (table, cols) + ",".join(values)

    create_table_if_not_exists(conn, table, df)

    try:
        cursor.execute(query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    cursor.close()

def table_exists(conn, table):
    cur = conn.cursor()
    cur.execute(f"select exists(select * from information_schema.tables where table_name='{table}')")
    exists = cur.fetchone()[0]
    cur.close()
    return exists

def create_table_if_not_exists(conn, table, df):
    # First check that the table doesn't exist
    curs = conn.cursor()
    cols = df.columns
    dtypes = [translate_dtype[str(x)] for x in df.dtypes]
    col_create = zip(cols, dtypes)
    cols = ','.join([i + ' ' + j for i, j in col_create])
    try:
        curs.execute(f"CREATE TABLE IF NOT EXISTS public.{table} ({cols})")
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        curs.close()
        return 1
    curs.close()
    return