import psycopg2
from psycopg2.extras import execute_values
import subprocess

# Connect to DB
print("--------------------\n--------------------\n--------------------\n--------------------Starting...")
conn = psycopg2.connect(
    dbname="gis",
    user="gis",
    password="gis",
    host="cs-u-spatial-406.cs.umn.edu",  # or wherever your DB is hosted
    port=5432
)
cur = conn.cursor()
print("\tConnection established")



# Standardize file
interim = subprocess.run([
    "python", "file_upload.py",
    "--filetype", "parquet",
    "--input", "data/Jakarta_subset3.parquet",
    "--id", "trj_id",
    "--t", "pingtimestamp",
    "--geometry", "point",
    "--lat", "rawlat",
    "--lon", "rawlng",
    "--output", "jakarta_inf_metadata.csv"
], capture_output=True, text=True)

print("STDOUT:", interim.stdout)
print("STDERR:", interim.stderr)

# Compute metadata
result = subprocess.run([
    "python", "get_metadata.py",
    "--input", "jakarta_inf_metadata.csv"
], capture_output=True, text=True)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)