import pandas as pd
import numpy as np

import datetime
from sklearn.cluster import KMeans
import osmnx as ox

class DataPartition:
    def __init__(self, df):

        # CONFIGS
        self.max_time_diff = pd.Timedelta(minutes=2)
        self.k = 4
        self.network_type = 'drive'
        self.buffer = 0.0005    # 50m buffer
        self.round_to = 6

        # DATASET
        self.traj_ids = df['traj_id'].to_list()
        self.timestamps = self.get_timestamp_list(col=df['timestamp'])
        self.latitudes = df['latitude'].to_list()
        self.longitudes = df['longitude'].to_list()

        self.num_points = len(df)

        # INDEXES
        self.time_group_idx, self.time_idx_exp, self.speeds = self.get_time_group_idx()
        self.space_group_mbrs, self.space_group_idx = self.get_space_groups()

        # DATA SUBSETS
        self.lats_in_time_groups = np.split(self.latitudes, self.time_group_idx)
        self.lons_in_time_groups = np.split(self.longitudes, self.time_group_idx)

    def print_group_indices(self):

        print(f"time group index: {self.time_group_idx}\n")
        print(f"time index expanded: {self.time_idx_exp}\n")
        print(f"space group MBRs: {self.space_group_mbrs}\n")
        print(f"space group index: {self.space_group_idx}\n")

        return

    def get_timestamp_list(self, col):
        """Convert timestamps to datetime objects"""
        dt = pd.to_datetime(col, errors = 'coerce')
        return dt.to_list()

    def get_time_group_idx(self):
        """
        Return:
            List of index intervals defining each time group
                All points in a time group are within the self.max_time_diff time difference
                All points in a time group are from the same trajectory
            Estimated speed at each point
                The last two points are given the same speed
        """
        time_group_intervals = []
        time_intervals_expanded = []
        time_diff = []
        dist_diff = []
        s = 0
        group_time_diff = pd.Timedelta(minutes=0)
        prev_id = self.traj_ids[0]
        prev_lat = self.latitudes[0]
        prev_lon = self.longitudes[0]
        prev_timestamp = self.timestamps[0]

        for i in range(1, (self.num_points-1)):

            cur_id = self.traj_ids[i]
            cur_lat = self.latitudes[i]
            cur_lon = self.longitudes[i]
            cur_timestamp = self.timestamps[i]

            if cur_id == prev_id:
                time_diff.append((cur_timestamp - prev_timestamp).total_seconds())
                dist_diff.append(round(np.linalg.norm(np.array([cur_lon, cur_lat], dtype=np.float32) 
                                                      - np.array([prev_lon, prev_lat], dtype=np.float32)), 
                                       self.round_to))
                # print(f"TIME D \t{(cur_timestamp - prev_timestamp).total_seconds()}")
                # print(f"DIST D \t{}")
                group_time_diff = cur_timestamp - self.timestamps[s]
                if (group_time_diff > self.max_time_diff):
                    time_group_intervals.append((s,i))
                    time_intervals_expanded.append(np.arange(s,i))
                    s = i
            else:
                # last two points of prev trajectory get same speed
                time_diff.append(time_diff[-1])
                dist_diff.append(dist_diff[-1])
                prev_id = cur_id
            
            prev_lat = cur_lat
            prev_lon = cur_lon
            prev_timestamp = cur_timestamp

        # last index - check if ids of last two points are the same, add correct ids to lists
        last_i = self.num_points - 1    # == i + 1
        cur_id = self.traj_ids[last_i]

        if cur_id != prev_id:   # corner case - last point not in same trajectory as penultimate point
            # add last group from loop
            time_group_intervals.append((s, i))
            time_intervals_expanded.append(np.arange(s, last_i))    # need last_i == i + 1 because we want [s, i]
            time_diff.append(time_diff[-1])
            dist_diff.append(dist_diff[-1])

            # add last point
            time_group_intervals.append(last_i, last_i)
            time_intervals_expanded.append(np.arange(last_i, self.num_points))
            time_diff.append(None)
            dist_diff.append(None)

        else:   # standard case - last point in same trajectory as penultimate point
            time_group_intervals.append((s, last_i))
            time_intervals_expanded.append(np.arange(s, self.num_points))

            cur_lat = self.latitudes[last_i]
            cur_lon = self.longitudes[last_i]
            time_diff.append((self.timestamps[last_i] - self.timestamps[i]).total_seconds())
            dist_diff.append(round(np.linalg.norm(np.array([cur_lon, cur_lat], dtype=np.float32) 
                                                  - np.array([prev_lon, prev_lat], dtype=np.float32)), 
                                   self.round_to))

        indices = [end for _, end in time_group_intervals[:-1]]  #TODO: make this the time_group_idx if we only need these vals   

        dt = [d/t for d, t in zip(dist_diff,time_diff)]
        dt.append(dt[-1])   # last two points have the same value

        return indices, time_intervals_expanded, dt
    
    def get_extrema(self):
        """
        CALLED BY GET_SPACE_GROUPS

        Get the min/max lat/long of the first and last points in a time group
        Return [min lat, max lat] , [min_long, max_long]
        """
        # indices = [end for _, end in self.time_group_idx[:-1]]  #TODO: make this the time_group_idx if we only need these vals   

        # [time group 1 [first point lat/lon, last point lat/lon], time group 2 [ first point, last point], ...]
        first_last_lats = np.matrix([[l[0], l[-1]] for l in np.split(self.latitudes, self.time_group_idx)])
        first_last_lons = np.matrix([[l[0], l[-1]] for l in np.split(self.longitudes, self.time_group_idx)])
        
        sorted_lats = np.asarray(np.sort(first_last_lats, axis=1))
        sorted_lons = np.asarray(np.sort(first_last_lons, axis=1))

        return sorted_lats, sorted_lons

    def get_bbox(self, lat, lon):
        """
        CALLED BY GET_SPACE_GROUPS

        Given a space group label, get all the lat/lon values [time group values] that are in that space group
            (lat/lon are list of [min val, max val] of each time group, and you mask by space group label i)
        Return (min_long, min_lat, max_long, max_lat) == (left, bottom, right, top)
        """
        w_min_lon = min(lon) - self.buffer
        s_min_lat = min(lat) - self.buffer
        e_max_lon = max(lon) + self.buffer
        n_max_lat = max(lat) + self.buffer
        return (w_min_lon, s_min_lat, e_max_lon, n_max_lat)

    def get_space_groups(self):
        """
        NOTES:  Number of space groups is the number of road networks
                We want to get the smallest number of road networks that are small enough
                to make map matching fast, so we cluster all the MBRs into k groups

        Return:
            space_groups: mbr of each space group (there are k groups)
                mbr = [min_long, min_lat, max_long, max_lat] == [left, bottom, right, top]
            labels: list of corresponding space group index for each time group
                index is the index of the space group each time group belongs in
        """
        sorted_lats, sorted_lons = self.get_extrema()
        mbrs = np.hstack([sorted_lats, sorted_lons])

        centroids = [( (x1+x2)/2, (y1+y2)/2 ) for y1, y2, x1, x2 in mbrs]
        kmeans = KMeans(n_clusters = self.k).fit(centroids)
        labels = kmeans.labels_

        space_groups = []
        for i in range(self.k):
            space_groups.append(self.get_bbox(lat=sorted_lats[i].tolist(), lon=sorted_lons[i].tolist()))

        return space_groups, labels
    
    def get_road_network(self, mbr):
        """
        CALLED BY MAP_MATCH

        Return OSM road network within the given space group mbr
        """
        g = ox.graph_from_bbox(bbox=mbr, network_type=self.network_type, 
                               retain_all=True, truncate_by_edge=True, simplify=False)
        # return ox.projection.project_graph(g)
        return g

    def get_all_space_group_points(self, i):
        """
        CALLED BY MAP_MATCH

        Given a space group label
        Return all the point coordinates that correspond to that space group as a list
            need to return lat and long separately because osm.nearest_edges takes them separately
            if it is no longer needed, can do them all at once:
                points_in_space_group = [time_group for time_group, space_idx in zip(self.points_in_time_groups, self.space_group_idx) if space_idx == i]
                return np.vstack(points_in_space_group).tolist()
        """
        lats_in_space_group = np.hstack([time_group for time_group, space_idx in zip(self.lats_in_time_groups, self.space_group_idx) if space_idx == i]).tolist()
        lons_in_space_group = np.hstack([time_group for time_group, space_idx in zip(self.lons_in_time_groups, self.space_group_idx) if space_idx == i]).tolist()
        indices = np.hstack([time_idx for time_idx, space_idx in zip(self.time_idx_exp, self.space_group_idx) if space_idx == i]).tolist()
        return lats_in_space_group, lons_in_space_group, indices

    def map_match(self):
        """
        CALLED BY GET_POINTS_INFO

        For each of the k space groups:
            get all points in that group, 
            map match all points using that road network
        Return the matching edges and distance from edge for each point
            in sorted order (sorted like original df: by traj_id and timestamp)
        """
        #TODO maybe rewrite with zip
        edges = []
        distances = []
        u_distances = []
        v_distances = []
        indices = []    # original point indices in order of space group
        for i in range(self.k):

            Y, X, all_t_idx_for_s = self.get_all_space_group_points(i)
            G = self.get_road_network(self.space_group_mbrs[i])
            e, d = ox.distance.nearest_edges(G, X, Y, return_dist=True)

            u_dist = [round(np.linalg.norm(np.array([G.nodes[row[0]]['x'], G.nodes[row[0]]['y']]) 
                                           - np.array([x, y])), 
                            self.round_to) for row, x, y in zip(e, X, Y)]
            
            v_dist = [round(np.linalg.norm(np.array([G.nodes[row[1]]['x'], G.nodes[row[1]]['y']]) 
                                           - np.array([x, y])), 
                            self.round_to) for row, x, y in zip(e, X, Y)]

            edges.append(e)
            distances.append(d)
            u_distances.append(u_dist)
            v_distances.append(v_dist)
            indices.append(all_t_idx_for_s)

        edges_list = np.hstack(edges).tolist()
        distance_list = np.hstack(distances).tolist()
        u_dist_list = np.concatenate(u_distances).tolist()
        v_dist_list = np.concatenate(v_distances).tolist()
        indices_list = np.hstack(indices).tolist()


        paired_e = list(zip(indices_list, edges_list))
        sorted_edges = [value for _, value in paired_e]

        paired_d = list(zip(indices_list, distance_list))
        sorted_distances = [value for _, value in paired_d]

        paired_u_dists = list(zip(indices_list, u_dist_list))
        sorted_u_dists = [value for _, value in paired_u_dists]

        paired_v_dists = list(zip(indices_list, v_dist_list))
        sorted_v_dists = [value for _, value in paired_v_dists]

        return sorted_edges, sorted_distances, sorted_u_dists, sorted_v_dists
    
    def get_point_info(self):
        """
        Return an np array with all the points in df 
            ordered by their trajectory ids and timestamp.

            Currently array only has trajector id, speed, edge, 
            and distance from edge of each point
        """
        edges, distances, u_dist, v_dist = self.map_match()
        s, d, k = zip(*edges)
        base_df = np.stack([self.traj_ids, self.speeds, s, d, k, u_dist, v_dist, distances]).transpose()
        return base_df