import subprocess
import time

# Standardize file
# interim = subprocess.run([
#     "python", "file_upload.py",
#     "--filetype", "parquet",
#     "--input", "/home/spatialuser/websites/openmetadata/modules/MetadataInference/data/Jakarta_subset.parquet",                         # "/home/spatialuser/websites/openmetadata/modules/MetadataInference/data/Jakarta_subset2.parquet",
#     "--id", "trj_id",
#     "--t", "pingtimestamp",
#     "--geometry", "point",
#     "--lat", "rawlat",
#     "--lon", "rawlng",
#     "--output", "jakarta_standardized.csv"
# ], capture_output=True, text=True)

# print("STDOUT:", interim.stdout)
# print("STDERR:", interim.stderr)
print("\n\n% % % % % % % % % % %\n")

time_str = time.strftime("%Y%m%d-%H%M%S")
# Compute metadata
result = subprocess.run([
    "python", "get_metadata.py",
    "--input", "jakarta_mini_standardized.csv",
    "--output", f"jakarta_mini_inf_metadata_{time_str}.csv"
], capture_output=True, text=True)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)