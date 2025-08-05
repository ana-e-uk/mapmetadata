import numpy as np

class Edge:
    def __init__(self, u, v, k):

        # GENERAL INFO
        self.u = u
        self.v = v
        self.k = k
        self.prev_p = None
        self.count = 0

        # SPEED VALS
        self.min_s = np.inf
        self.max_s = -1
        self.q1 = 0
        self.q2 = 0
        self.q3 = 0

        # TO CALCULATE METADATA
        self.u_to_v_count = 0
        self.v_to_u_count = 0
        self.max_dist = 0

        # METADATA
        self.oneway = True
        self.expected_speed = None
        self.speed_limit = None

    def update(self, cur_p):
        """Update edge statistics using current point"""

        # speed extrema
        s = cur_p["speed"]
        self.min_s = min(self.min_s, s)
        self.max_s = max(self.max_s, s)

        # speed quantiles
        #   computed using previous quantile value and new speed value
        #   this got similar results to updating using the min and max alongise points
        self.q1 = np.quantile([self.q1,s],q=0.25)
        self.q2 = np.quantile([self.q2,s],q=0.5)
        self.q3 = np.quantile([self.q3,s],q=0.75)

        # max distance used to calculate number of lanes
        self.max_dist = max(self.max_dist, cur_p["dist"])

        # direction of trajectory
        # check if points are from same trajectory and w/in 2 mins
        if self.prev_p is not None:
            if cur_p["traj_id"] == self.prev_p["traj_id"]:
                if(cur_p["timestamp"] - self.prev_p["timestamp"]).total_seconds() < 120:
                    
                    # compute direction
                    if cur_p["u_d"] < self.prev_p["u_d"]:
                        if cur_p["v_d"] > self.prev_p["v_d"]:
                            self.v_to_u_count += 1
                            print(f"\tv to u")
                        else:
                            self.u_to_v_count += 1
                            print(f"\tu to v")
                    else:
                        if cur_p["v_d"] > self.prev_p["v_d"]:
                            self.v_to_u_count += 1
                            print(f"\tv to u")
                        else:
                            self.u_to_v_count += 1
                            print(f"u to v")
        self.prev_p = cur_p

        # updating number of points for edge
        self.count += 1

    def get_oneway(self):
        """Use u to v/ v to u counts to determine if edge is a oneway"""
        if (self.u_to_v_count == 0) or (self.v_to_u_count == 0):
            assert self.oneway == True
        else:
            self.oneway = False
        return

    def get_expected_speed(self):
        """Expected speed = q2"""
        self.expected_speed= round(self.q2,3)

    def get_speed_limit(self):
        """
        Return guess of legal speed limit
            Guess: the closest multiple of ten greater than the maximum speed observed
        """
        t = np.trunc(self.max_s/10)
        self.speed_limit = int((t*10) + 10)


class EdgesSet:
    def __init__(self):
        self.edges = {}  # key: (u, v, k) -> value: Edge object

    def update_edge(self,cur_p):
        """
        Given an edge index (u, v, k) and the current point

            - Add edge to set if it is not in it yet
            - Update edge statistics given current point cur_p
        """
        idx = (cur_p["u"], cur_p["v"], cur_p["k"])
        if idx not in self.edges:
            print("new edge!")
            self.edges[idx] = Edge(cur_p["u"], cur_p["v"], cur_p["k"])
        self.edges[idx].update(cur_p)

    def get_edge(self, u,v,k):
        """Return edge at given index"""
        return self.edges.get((u,v,k), None)
    
    def get_all_idx(self):
        return self.edges.keys()
    
    def compute_metadata(self, u, v, k):
        
        edge = self.edges[(u,v,k)]

        edge.get_oneway()
        edge.get_expected_speed()
        edge.get_speed_limit()

        return edge.oneway, edge.expected_speed, edge.speed_limit 