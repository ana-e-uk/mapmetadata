import subprocess
import time

# Standardize file
interim = subprocess.run([
    "python", "file_upload.py",
    "--filetype", "parquet",
    "--input", "/home/spatialuser/websites/openmetadata/modules/MetadataInference/data/Jakarta_subset2.parquet",    # Jakarta_subset (13M) subset2 (136M) subset3 (1.6G) sorted (1.8G)
    "--id", "trj_id",
    "--t", "pingtimestamp",
    "--geometry", "point",
    "--lat", "rawlat",
    "--lon", "rawlng",
    "--output", "j2_standardized.csv"
], capture_output=True, text=True)

print("STDOUT:", interim.stdout)
print("STDERR:", interim.stderr)
print("\n\n% % % % % % % % % % %\n")

time_str = time.strftime("%m-%d-%H:%M")
# Compute metadata
result = subprocess.run([
    "python", "get_metadata.py",
    "--input", "j2_standardized.csv",
    "--output", f"j2_{time_str}.csv"
], capture_output=True, text=True)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)