### `file_upload.py` File Upload and Standardization

#### Types of Files

1. ESRI shapefile ([docs](https://www.esri.com/content/dam/esrisites/sitecore-archive/Files/Pdfs/library/whitepapers/pdfs/shapefile.pdf))
    * Main and index file, dBASE table
    * Non-topological geometry and attribute info 
    * Stores GPS trajectory as a feature (?)
    * FastMapMatching accepts this type of file

2. CSV trajectory file 
    * Each row stores: trajectory, ID, geometry (WKT linestring format), and timestamp
    * Column names specified by user: ID, geometry (linestring), longitude (0 if first in tuple, 1 else), latitude (0 or 1), timestamp 
    * FastMapMatching expects this type of file

3. CSV point file
    * Each row stores: ID, x, y, timestamp
    * Sorted by ID and timestamp
    * Column names specified by user: ID, geometry (point), longitude, latitude, timestamp

4. GPX file
    * OSM requires this file for GPS traces
    * Stores a latitude and longitude value for each point
    * Stores timestamps in UTC

5. KML file
    * Extended from XML file format
    * Saves latitude, longitude, altitude for each point

6. GeoJSON
    * Extended from JSON file format
    * Saves point, linestring, polygon etc. geometries

#### Jakarta Dataset Example
To upload and standardize file run:

    python file_upload.py   --filetype parquet --input data/Jakarta_subset3.parquet  --id trj_id --t pingtimestamp --geometry point  --lat rawlat --lon rawlng --output jakarta_inf_metadata.csv


#### GPS File List
* Porto
* Beijing
* Jakarta
* Pakistan
* San Francisco
* [US](https://catalog.data.gov/dataset/?tags=taxis) taxi datasets in Chicago.
* [US](https://www.fhwa.dot.gov/policyinformation/travelmonitoring.cfm) Travel monitoring and traffic volume.
* [US](https://catalog.data.gov/dataset/taxi-trips-in-2024) taxi datasets in DC.
* [US](https://catalog.data.gov/dataset/2023-for-hire-vehicles-trip-data) for hire vehicles trip data in New York.
* [Czech Republic](https://datasetsearch.research.google.com/search?ref=TDJjdk1URjBOV1J3TTJNeWNRPT0sTDJjdk1URjNhR1k0WW10b2JBPT0sTDJjdk1URnNaR3gwWm5SNGNRPT0%3D&query=GPS%20trip&docid=L2cvMTF3dHBfazBkcA%3D%3D) kaggle vehicle trip data
* [Unknown](https://datasetsearch.research.google.com/search?src=0&query=GPS%20trip&docid=L2cvMTF3eGJocTh2bA%3D%3D) trajectory data on kaggle used for driving style IBM-China research.
* [Private vehicle](https://zenodo.org/records/4449671) dataset with engine status and and angle.
* [Kaggle](https://datasetsearch.research.google.com/search?src=0&query=GPS%20trip&docid=L2cvMTFxMXY4d3Q3Nw%3D%3D) search to look at more later.


**Resources**
FastMapMatching GPS data input ([docs](https://fmm-wiki.github.io/docs/documentation/input/))


### `get_metadata.py` Map Matching and Metadata Computation

When a new file is uploaded and standardized, we can update the metadata. The metadata (saved in a large table for every road and intersection) for a road can be updated if a point corresponds to (is map matched to) that road. Once we go through all the points in the new file, we will have updated the relevant metadata.

#### General Steps
1. Divide points into time intervals smaller than **max_time_diff** to create *time groups*
2. Compute speed at each point
3. Divide *time groups* spatially into **k** *space groups* where each *space group* contains all its corresponding *time groups*
4. Map match all points to a road network and save distance from point to edge
5. Compute distance from point to both edge nodes
6. For each point, update the corresponding edge information
7. For each edge that was updated, compute the metadata

#### Time Complexity

#### Description
(Describes the functions/classes that do each step)

<!-- ### Map Matching Notes
Does map matching need to be completely accurate or can we just estimate it?
1. split roads into layers (residential, highway, etc.)
2. match gps trajectories to all roads within a certain threshold distance (such that highway segments are not chosen when residential segments are more likely, maybe by classifying the trajectory (many turns = city/residential))
3. use the speed/data from that point on all those road segments, weighted by their distance from the point, so points closer to a road have more influence on the road's metadata, maybe with Sparkle distance formula...

Points have to be matched to a road, to have all the points in a road you have to go through all the points, then go through the points again, so it would be better to update the metadata while map matching. -->

#### Metadata Computed

**DONE:**
* **Driving direction** - direction points are traveling on a road 
* **Number of lanes** - estimated number of lanes based on the width of a road calculated from the distance (UNITS km/h) of points to the center 
* **Point count** - number of points matched to that road

**WORK IN PROGRESS**
* **Speed info** - UNITS km/h
* **Possible turns** - list of neighboring edges that points have turned on from each road node
* **Intersection type** - category of intersection from: *roundabout, controlled, uncontrolled*
* **Road type** - category of road from: *highway, city, residential*
* **Parking type** - category of parking allowed on a road from: *none, short-term, long-term*

#### Jakarta Dataset Example
To get the metadata from the Jakarta dataset, run:

    python get_metadata.py --input jakarta_inf_metadata.csv

To compare the inferred metadata with the OSM metadata values, run:

    python compare_metadata.py

<!-- ### Calculate Metadata Notes

We can either:
1. get the map matching, then for each edge, calculate metadata, then every new point updates edge data. Metadata that requires all the information needs to be calculated at the end.
2. for each point, map match and calculate metadata at the same time, updating edge each time -->


### Visualize Metadata

### Download Metadata