import argparse
import sys

import pandas as pd

from DataPartition import DataPartition
from EdgeSet import EdgeSet

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
        partition = DataPartition(df=df)
        point_data = partition.get_point_info()
        point_df = pd.DataFrame(point_data, columns=["traj_id", "timestamp", "speed", "u", "v", "k", "u_d", "v_d", "dist"])

        es = EdgeSet()
        for p in point_df:
            es.update_edge(p)
        
        for e in es.get_all_idx():
            e_metadata = es.compute_metadata(e["u"], e["v"], e["k"])
            

    except Exception as e:
        print(f"Failed to process the file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()