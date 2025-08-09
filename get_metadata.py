import argparse
import sys

import pandas as pd
import time

from DataPartition import DataPartition
from EdgeSet import EdgeSet

def get_outfile_name(file_out):
    if file_out is not None:
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
        start_time = time.time()
        partition = DataPartition(df=df)
        partition_time = time.time()
        point_data = partition.get_point_info()
        partition_2_time = time.time()

        es = EdgeSet()
        for p in point_data: 
            es.update_edge(p)
        edge_time = time.time()

        edge_metadata = []
        for e in es.get_all_idx():
            edge_metadata.append(es.compute_metadata(e[0], e[1], e[2]))
        metadata_time = time.time()
        
        metadata_df = pd.DataFrame(edge_metadata, columns=["osmid", "inf_oneway_direction", "e_speed", "speed_limit"])
        csv_name = get_outfile_name(args.output)
        metadata_df.to_csv(csv_name, index=False)
        print(f"GET METADATA TIMES:\n\tinitialize partition: {partition_time - start_time}\tget point data: {partition_2_time - partition_time} \t(total): {partition_2_time - start_time}\n\tedge update: {edge_time - partition_2_time}\n\tmetadata for edges: {metadata_time - edge_time}\n")
        print(f"\tTOTAL: minutes {round((metadata_time - start_time)/60, 3)}")
        print("\n############################ END ################################")
    except Exception as e:
        print(f"Failed to process the file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()