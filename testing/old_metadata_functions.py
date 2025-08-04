import argparse
import numpy as np
from haversine import haversine, Unit
import pandas as pd
import sys
import osmnx as ox
import geopandas as gpd

def get_oneway_bool(e):
    # oneway is false by default
    n_a_b = e["driv_dir"][1]
    n_b_a = e["driv_dir"][0]
    sum_dir = n_a_b + n_b_a
    oneway = True if ((n_b_a == 0 or n_a_b == 0) and (sum_dir > 3))  else False
    if (not oneway and (min(n_b_a, n_a_b) < 10)):
        # check if there is one outlier s.t. oneway should be true
        if max(n_b_a, n_a_b) > 5*(min(n_b_a, n_a_b)):
            oneway = True

    return oneway

def get_driving_direction(p1, p2, n_A, n_B):
    
    p1_to_A = haversine(p1, n_A, unit=Unit.METERS)
    p1_to_B = haversine(p1, n_B, unit=Unit.METERS)

    p2_to_A = haversine(p2, n_A, unit=Unit.METERS)
    p2_to_B = haversine(p2, n_B, unit=Unit.METERS)
    
    # case where points are within 1 meter of eachother
    if (np.abs(p1_to_A - p2_to_A) < 1):
        if p1_to_A < 5: # points are within 5 meters of A
            return 2
    elif (np.abs(p1_to_B - p2_to_B) < 1):
        if p1_to_B < 5: # points are within 5 meters of B
            return 3
    
    else:
        if p1_to_A < p2_to_A:
            if p1_to_B < p2_to_B:
                return None
            else:
                return 1    # A -> B
        else:
            if p1_to_B > p2_to_B:
                return None
            else:
                return 0    # B -> A
    

def get_road_type(e):

    '''
    Road types: 1 - highway
                2 - city
                3 - residential
    '''

    maxspeed = e["maxspeed"]
    num_lanes = int(round(np.divide(e["width"], 12)))
    count = e["count"]
    
    if maxspeed >= 40:
        return 1
    else:
        if num_lanes > 2:
            return 2
        if count > 10:
            return 3

def map_match_point(G_proj, lat, long):
    """Match point to a road in road network and return road id"""
    edge = ox.nearest_edges(G_proj, lat, long, return_dist=True)
    return edge

def compute_metadata(df):
    """Compute the metadata for each point"""
    df = df.sort_values(['traj_id', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # PRECOMPUTATIONS

    # Shift lat/lon/timestamp for each traj_id
    df['next_lat'] = df.groupby('traj_id')['latitude'].shift(-1)
    df['next_lon'] = df.groupby('traj_id')['longitude'].shift(-1)
    df['next_time'] = df.groupby('traj_id')['timestamp'].shift(-1)
    # Compute distance btwn points
    df['distance_m'] = df.apply(lambda row: haversine(
        (row['latitude'], row['longitude']),
        (row['next_lat'], row['next_lon']),
        unit=Unit.METERS
    ) if not pd.isnull(row['next_lat']) else 0, axis=1)
    # Compute time delta in seconds
    df['time_delta'] = (df['next_time'] - df['timestamp']).dt.total_seconds()
    # Speed m/s
    df['speed_mps'] = df['distance_m'] / df['time_delta']

    # UPDATE METADATA

    # Get the graph from the bounding box of the trajectory and project both to the same CRS
    lat_col = df['latitude']
    long_col = df['longitude']
    n = max(lat_col)
    s = min(lat_col)
    e = max(long_col)
    w = min(long_col)
    network_type = "drive"
    G = ox.graph_from_bbox([n, s, e, w], network_type=network_type, simplify=False, retain_all=True, truncate_by_edge=False)
    G_proj = ox.project_graph(G)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(long_col, lat_col), crs='EPSG:4326')
    gdf_proj = gdf.to_crs(G_proj.graph['crs'])

    long_col_proj = gdf_proj.geometry.x
    lat_col_proj = gdf_proj.geometry.y
    edges = ox.nearest_edges(G_proj, lat_col_proj, long_col_proj, return_dist=True)
    #TODO: save edges only to col
    # df['edges'] = 
    #TODO: save distance only to col
    # df['edge_distance'] = 

    # for lat, long in zip(lat_col, long_col):
    #     edge = map_match_point(G_proj, lat, long)

    #TODO: call the actual edges dict/metadata info?? or wait to update at the end
    edges_dict = {}

    for i, e in zip(len(df), edges):

        row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        if e not in edges_dict.keys():
            edges_dict[e] = {"maxspeed": int(row["speed_mps"]),
                            "driv_dir": [int(0), int(0)], # [ B->A, A->B ] so index[0]=count(B->A), index[1]=count(A->B)
                            "turns": {"A": [], "B": []},
                            # "width": int(2*row["edge_distance"]),
                            "width": 0,
                            "i_type": [int(0), int(0)],   # [B, A]
                            "r_type": int(-1),
                            "p_type": [int(0), int(0)],   # [ left A-B, right A-B ] 
                            "count": int(0)
                            }
        
        cur_e = edges_dict[e]

        parking = True if row["speed"] < 1.5 else False
        cur_e["maxspeed"] = max(cur_e["maxspeed"], row["speed"])
        cur_e["width"] = max(cur_e["width"], 2*row["edge_distance"])
        cur_e["count"] += 1
        
        if row["trajectory_id"] == next_row["trajectory_id"]:    # Same trajectory

            if e == next_row["edge_id"]: # Same edges

                direction = get_driving_direction(p1=(row["latitude"], row["longitude"]), 
                                                p2=(next_row["latitude"], next_row["longitude"]),
                                                    n_A=(ast.literal_eval(row["node_A_coords"])[1], ast.literal_eval(row["node_A_coords"])[0]), 
                                                    n_B=(ast.literal_eval(row["node_B_coords"])[1], ast.literal_eval(row["node_B_coords"])[0]))
                
                if direction:
                    if direction < 2: # 0 or 1
                        cur_e["driv_dir"][direction] += 1
                        if parking:
                            cur_e["p_type"][direction] += 1
                    elif direction == 2: # A
                        cur_e["i_type"][1] += 1
                    elif direction == 3: # B
                        cur_e["i_type"][0] += 1

            else:
                p1_ab = row["edge_id"]
                p2_ab = next_row["edge_id"]
                p1_a = p1_ab[0]
                p1_b = p1_ab[1]

                if (p1_a in p2_ab):  # Adjacent edge
                    if p2_ab in cur_e["turns"]["A"]:
                        pass
                    else:
                        cur_e["turns"]["A"].append(p2_ab)
                    cur_e["driv_dir"][0] = 1    # B -> A
                elif (p1_b in p2_ab):   # Adjacent edge
                    if p2_ab in cur_e["turns"]["B"]:
                        pass
                    else:
                        cur_e["turns"]["B"].append(p2_ab)
                    cur_e["driv_dir"][1] = 1    # A -> B
                
                else:   # No adjacent edges
                    pass

        else:   # Different trajectory
            pass
        
        # Get intersection, road, and parking types
        if cur_e["count"] > 1:
            cur_e["r_type"] = get_road_type(cur_e)

        # Save updated values of edge e
        edges_dict[e] = cur_e

    # Estimated speed limit


    # Maximum speed limit

    # Driving direction

    # Possible turns

    # Number of lanes

    # Intersection type

    # Road type

    # Parking type

    # Point count
    
    return df 

def main():
    parser = argparse.ArgumentParser(description="Process trajectory file to update road metadata.")

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the trajectory file'
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        
        compute_metadata(df)
    except Exception as e:
        print(f"Failed to process the file: {e}")
        sys.exit(1)


class MetadataInference:
    def __init__(self):
        
        self.edge_stats = []

    def speed_limit(self):
        """ Return estimated speed limit"""

    def expected_speed(self):
        """ Return estimated speed limit"""


    def edge_stats(self):
        """
        Updates edge stats
            - boxplot data: floats of quartiles and outliers
            - start to end: 0 if no trajectories go from start to end nodes 1 else
            - end to start: 0 if no trajectories go from end to start nodes 1 else
            - s: list of edges a trajectory turned onto from start node
            - e: list of edges a trajectory turned onto from end node
            - max d: float of largest distance from a point to its edge
        """
    

    def get_metadata(self):
        """
        Calls other functions to get all the metadata.

        Returns metadata for edge that is matched to at least one point in df
            - max speed: estimated legal speed limit
            - expected speed: mode of observed speeds
            - oneway: 0 if a road is two-directional, 1 else
            - number of lanes: number of lanes of an edge
            - road type: whether road is a highway, city, residential
            - parking type: no parking or loading, loading, parking

        NOT IMPLEMENTED YET
            - turns: possible turns from node
            - intersection type: stop sign, traffic light, yield
        """
