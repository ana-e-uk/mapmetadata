# Extracting Map Metadata from GPS Trajectories

### Create Environment
1. In the terminal, run:
    
        conda env create -f portable_env.yml

    * To activate environment: **conda activate mapedia_env**
    * To deactivate environment: **conda deactivate**

### File Upload and Standardization

1. In the terminal, run:

        python file_upload.py --filetype --input --id --t --geometry --lat --lon

    where
    * --filetype: one of .csv, .json, .xlsx, .parquet, .feather file types
    * --input: the directory path of the file to upload
    * --id: trajectory ID column name 
    * --t: date time column name
    * --geometry: 'point' if coordinates are in rows or 'linestring' if coordinates are in linestring object
    * --lat: latitude column name or index of latitude coordinate (0 or 1)
    * --long: longitude column name or index of longitude coordinate (0 or 1)
    
2. Data is standardized to have the same column names: **traj_id, latitude, longitude, timestamp**, be in a trajectory point format, where each row is a point in a trajectory, and are all saved as a .csv file.

TODO:
* get input parameters from the website
* figure out where input file is saved, and where to save the tempory standardized file [ALL IN /data FOLDER]

### Map Matching and Metadata Computation

1. In terminal, run:
    
        python get_metadata.py --input
    
    where
    * --input: the directory path of the standardized file

2. The metadata will be computed as follows using the points in the file:
    * Using the **DataPartition** class:
        * The data will be divided into **k** spatial regions, and the speed of each point wil be calculated and saved.
        * Each point will be map matched using the OSM road network for its spatial region, and some information will be saved for each point:
            * Matched edge index *(u, v, k)*, distance of point to *u*, distance of point to *v*, distance to edge.
    * Using the **EdgeSet** class:
        * For each point, the corresponding edge will be updated with the information provided by that point.
    * The final metadata for each edge in the created *EdgeSet* will be computed.

Note: **Code_Documentation.md** holds documentation on the steps above

TODO:
* check the sorting in map match function works correctly
* rewrite functions that take too much time
* figure out best place to project points and graph
* check distance from edge returned by **nearest_edges** is perpendicular distance to edge
* make get_metadata start after standardization is done (maybe just call it when standardization is done so you don't have to be constantly checking for new files)

### Visualize Metadata

### Download Metadata
