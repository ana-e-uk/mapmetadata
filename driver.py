import subprocess

# Standardize file
interim = subprocess.run([
    "python", "file_upload.py",
    "--filetype", "parquet",
    "--input", "/home/spatialuser/websites/openmetadata/modules/MetadataInference/data/Jakarta_subset3.parquet",
    "--id", "trj_id",
    "--t", "pingtimestamp",
    "--geometry", "point",
    "--lat", "rawlat",
    "--lon", "rawlng",
    "--output", "jakarta3_standardized.csv"
], capture_output=True, text=True)

print("STDOUT:", interim.stdout)
print("STDERR:", interim.stderr)
print("\n\n% % % % % % % % % % %\n")

# Compute metadata
result = subprocess.run([
    "python", "get_metadata.py",
    "--input", "jakarta3_standardized.csv"
    "--output", "jakarta3_inf_metadata.csv"
], capture_output=True, text=True)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)