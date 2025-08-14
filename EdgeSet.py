import numpy as np
from collections import Counter

class Edge:
    def __init__(self, u, v, k):

        # GENERAL INFO
        self.u = u
        self.v = v
        self.k = k
        self.round_to = 2

        self.osmid = None
        self.osm_hway = None
        self.osm_maxspeed = np.nan
        self.osm_oneway = None
        self.osm_lanes = np.nan

        self.osmid_l = []
        self.osm_hway_l = []
        self.osm_maxspeed_l = []
        self.osm_oneway_l = []
        self.osm_lanes_l = []

        self.prev_p = None
        self.count = 0

        # SPEED VALS
        self.min_s = np.inf
        self.max_s = -1
        self.q1 = None
        self.q2 = None
        self.q3 = None
        self.four_points = False

        # TO CALCULATE ONEWAY, LANES
        self.u_to_v_count = None
        self.v_to_u_count = None
        self.max_dist = 0       # max_dist is in km

        # METADATA
        self.inf_oneway = None
        self.inf_expected_speed = None
        self.inf_speed_limit = None
        self.inf_lanes = None

    def get_direction_counts(self, cur_p):
        if(cur_p[1] - self.prev_p[1]).total_seconds() < 120:    # cur["timestamp"]  
            # compute direction
            if cur_p[11] < self.prev_p[11]:     # cur_p["u_dist"]
                if cur_p[12] > self.prev_p[12]: # cur_p["v_dist"]
                    if self.v_to_u_count:
                        self.v_to_u_count += 1
                    else:
                        self.v_to_u_count = 1
                else:
                    if self.u_to_v_count:
                        self.u_to_v_count += 1
                    else:
                        self.u_to_v_count = 1
            else:
                if cur_p[12] > self.prev_p[12]:
                    if self.v_to_u_count:
                        self.v_to_u_count += 1
                    else:
                        self.v_to_u_count = 1
                else:
                    if self.u_to_v_count:
                        self.u_to_v_count += 1
                    else:
                        self.u_to_v_count = 1
        return 
    
    def get_quantile_vals(self, s):
        if self.four_points:
            self.q1 = round(np.quantile([self.q1,s],q=0.25), self.round_to)
            self.q2 = round(np.quantile([self.q2,s],q=0.5), self.round_to)
            self.q3 = round(np.quantile([self.q3,s],q=0.75), self.round_to)
        elif self.q1 is None:
            self.q1 = s
        elif self.q2 is None:
            self.q2 = s
        elif self.q3 is None: 
            self.q3 = s
        else:
            self.four_points = True
            first_four = [self.q1, self.q2, self.q3, s]
            self.q1 = round(np.quantile(first_four, q=0.25))
            self.q2 = round(np.quantile(first_four, q=0.5))
            self.q3 = round(np.quantile(first_four, q=0.75))
    
    def update(self, cur_p):
        """Update edge statistics using current point"""

        # speed extrema
        s = cur_p[2] # cur_p["speed"]
        self.min_s = round(min(self.min_s, s), self.round_to)
        self.max_s = round(min(max(self.max_s, s),120), self.round_to)

        # max distance used to calculate number of lanes
        self.max_dist = max(self.max_dist, cur_p[13])    #cur_p["dist"])

        # direction from two points in same trajectory
        if self.prev_p is not None:
            if cur_p[0] == self.prev_p[0]:  # cur["traj_id"]
                self.get_direction_counts(cur_p)
            
        # speed quantiles
        self.get_quantile_vals(s)

        self.osmid_l.append(cur_p[3])
        self.osm_hway_l.append(cur_p[4])
        self.osm_maxspeed_l.append(cur_p[5])
        self.osm_oneway_l.append(cur_p[6])
        self.osm_lanes_l.append(cur_p[7])

        self.prev_p = cur_p

        # updating number of points for edge
        self.count += 1
    
    def get_osm_consensus(self):
        """Return majority osm values as final osm values"""
        self.osmid = Counter(self.osmid_l).most_common(1)[0][0]
        self.osm_hway = Counter(self.osm_hway_l).most_common(1)[0][0]
        self.osm_maxspeed = Counter(self.osm_maxspeed_l).most_common(1)[0][0]
        self.osm_oneway = Counter(self.osm_oneway_l).most_common(1)[0][0]
        self.osm_lanes = Counter(self.osm_lanes_l).most_common(1)[0][0]

        # print(f"\n\tGET_OSM_CONSENSUS:\nosmid list: {np.unique(self.osmid_l)}")
        # print(f"\nosm highway list: {np.unique(self.osm_hway_l)}")
        # print(f"\nosm maxspeed list: {np.unique(self.osm_maxspeed_l)}")
        # print(f"\nosm oneway list: {np.unique(self.osm_oneway_l)}")
        # print(f"\nosm lanes list: {np.unique(self.osm_lanes_l)}")

    def get_oneway(self):
        """Use u to v/ v to u counts to determine if edge is a oneway"""
        if self.u_to_v_count and self.v_to_u_count:

            sm = min(self.u_to_v_count, self.v_to_u_count)
            lg = max(self.u_to_v_count, self.v_to_u_count)
            ratio = sm/lg

            if (self.u_to_v_count == 0) or (self.v_to_u_count == 0):
                self.inf_oneway = True
                # TODO: this sets two-ways as oneways if we do not have enough samples
            elif ratio < 0.55:
                if sm + lg < 11:    # small counts are exception, not enough data to say if count imbalance is error
                    self.inf_oneway = False
                self.inf_oneway = True  # one direction has ~less than half the counts as the other
            else:
                self.inf_oneway = False 
        return

    def get_expected_speed(self):
        """Expected speed = q2"""
        self.inf_expected_speed= self.q2

    def get_speed_limit(self):
        """
        Return guess of legal speed limit
            Guess: the closest multiple of ten greater than the maximum speed observed
            Check: if the max value observed is 40km or greater than q3 value, use q3 value as max
        """
        if self.q3:
            if (self.max_s - self.q3) >= 40:
                t = np.trunc(self.q3/10)
            else:
                t = np.trunc(self.max_s/10)
        else:
            t = np.trunc(self.max_s/10)
        self.inf_speed_limit = int((t*10) + 10)

    def get_number_of_lanes(self):
        """
        Returns number of 12ft lanes that can fit w/in
            the largest distance from point to edge * 2
        """
        n_lanes = (np.ceil(self.max_dist*1000) * 2)/ 3.6  # avg lane width ~= 3.6 meters
        self.inf_lanes = int(max(np.ceil(n_lanes), 1))
    
    def update_number_of_lanes(self):
        """Handles cases where given speed/road metadata, you expect more than 1 lane"""
        pass

class EdgeSet:
    def __init__(self):
        # key: (u, v, k) -> Edge object
        self.edges = {}

    def _ensure_edge_exists(self, idx, cur_p):
        """
        Internal helper: create an Edge if it doesn't exist yet.
        idx: tuple (u, v, k)
        cur_p: current point (list/tuple)
        """
        if idx not in self.edges:
            self.edges[idx] = Edge(cur_p[8], cur_p[9], cur_p[10])

    def create_edge(self, cur_p):
        """
        Ensure the edge exists, then update it.
        """
        idx = (cur_p[8], cur_p[9], cur_p[10])
        self._ensure_edge_exists(idx, cur_p)
        self.edges[idx].update(cur_p)

    def get_edge(self, u, v, k):
        """Return edge at given index or None if not present."""
        return self.edges.get((u, v, k), None)

    def get_all_idx(self):
        """Return all edge keys."""
        return self.edges.keys()

    def compute_metadata(self, u, v, k):
        """Compute and return metadata for a given edge."""
        cur_e = self.edges[(u, v, k)]
        cur_e.get_osm_consensus()
        cur_e.get_oneway()
        cur_e.get_expected_speed()
        cur_e.get_speed_limit()
        cur_e.get_number_of_lanes()
        return (
            cur_e.u,
            cur_e.v,
            cur_e.k,
            cur_e.osmid,
            cur_e.inf_oneway,
            cur_e.osm_oneway,
            cur_e.inf_expected_speed,
            cur_e.inf_speed_limit,
            cur_e.osm_maxspeed,
            cur_e.inf_lanes,
            cur_e.osm_lanes,
            cur_e.osm_hway,
            cur_e.count,
            cur_e.min_s,
            cur_e.max_s,
            cur_e.q1,
            cur_e.q2,
            cur_e.q3,
            (cur_e.u_to_v_count, cur_e.v_to_u_count)
        )


# import numpy as np

# class EdgeSet:
#     def __init__(self):
#         self.edges = {}  # key: (u, v, k) -> value: Edge object
#         self.edge_osmids = []

#     def create_edge(self,cur_p):
#         """
#         Given an edge index (u, v, k) and the current point
#             - Add edge to set if it is not in it yet
#             - Update edge statistics given current point cur_p
#         """
#         idx = (cur_p[8], cur_p[9], cur_p[10])       # (cur_p["u"], cur_p["v"], cur_p["k"])
#         cur_osm_id = str(cur_p[3])

#         if cur_osm_id not in self.edge_osmids:
#             self.edges[idx] = Edge(cur_p[8], cur_p[9], cur_p[10])
#             self.edge_osmids.append(str(cur_osm_id))
#         self.edges[idx].update(cur_p)
#         return
    
#     def get_edge(self, u,v,k):
#         """Return edge at given index"""
#         return self.edges.get((u,v,k), None)
    
#     def get_all_idx(self):
#         return self.edges.keys()
    
#     def compute_metadata(self, u, v, k):
        
#         edge = self.edges[(u,v,k)]

#         edge.get_oneway()
#         edge.get_expected_speed()
#         edge.get_speed_limit()

#         return edge.edge.osmid, edge.inf_oneway, edge.osm_oneway, edge.inf_expected_speed, edge.inf_speed_limit, edge.osm_maxspeed