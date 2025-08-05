import argparse
import sys

import pandas as pd
import time

from DataPartition import DataPartition
from EdgeSet import EdgeSet

def get_outfile_name(file_out):
    if file_out != None:
        return file_out
    else:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        return f'{timestr}.csv'

def main():
    parser = argparse.ArgumentParser(description="Process trajectory file to update road metadata.")

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the trajectory file'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Optional output file'
    )

    args = parser.parse_args()

    try:

        df = pd.read_csv(args.input)
        partition = DataPartition(df=df)
        point_data = partition.get_point_info()
        point_df = pd.DataFrame(point_data, columns=["traj_id", "timestamp", "speed", "osmid", "u", "v", "k", "u_d", "v_d", "dist"])

        es = EdgeSet()
        for p in point_df:
            es.update_edge(p)
        
        edge_metadata = []
        for e in es.get_all_idx():
            edge_metadata.append(es.compute_metadata(e["u"], e["v"], e["k"]))

        edge_metadata_df = pd.DataFrame(edge_metadata)
        csv_name = get_outfile_name(args.output)
        edge_metadata_df.to_csv(csv_name, index=False)

    except Exception as e:
        print(f"Failed to process the file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()