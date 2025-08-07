import argparse
import pandas as pd
import sys
# from shapely import LineString
import time
import os

def check_file_saved(file_name):
    """Verify the csv file that was saved exists."""
    try:
        os.path.isfile(file_name)
    except Exception as e:
        print(f"Error: The file was not saved correctly and does not exist. Please upload again.\nError: {e}")
        sys.exit(1) # Exit with error code
    

def validate_columns(df, df_columns, user_columns, geom, file_out=None, linestring_col=""):
    """Validate that all user-specified columns exist in the dataframe."""
    #TODO: if a speed column is specified, standardize name
    #TODO: check the linestring tests
    #TODO: add directory for temporary file
    #TODO: add crs option for lat and long and convert to crs='EPSG:4326' here

    missing = [col for col in user_columns if col not in df_columns]
    if missing:
        print(f"Error: The following columns were not found in the file: {missing}")
        sys.exit(1)  # Exit with error code
    else:
        col_id = user_columns[0]
        col_t = user_columns[1]
        col_lat = user_columns[2]
        col_long = user_columns[3]

        # try:
        #     pd.to_datetime(df[col_t], errors='coerce')    # checking timestamps are date times
        # except:
        #     print(f"Error: the values in the timestamp column could not be converted to date times")
        #     sys.exit(1)
        if geom == "point":
            if df[col_lat].astype(float).all(): 
                pass    # latitude values are floats
            else:
                print("Error: Latitude column data type is not float")
                sys.exit(1)
            if df[col_long].astype(float).all(): 
                pass    # longitude values are floats
            else:
                print("Error: Longitude column data type is not float")
                sys.exit(1)
        else:   
            if isinstance(df[linestring_col][0][int(col_lat)], float): 
                pass # lat values are floats
            else:
                print("Error: Latitude column data type is not float")
                sys.exit(1)
            if isinstance(df[linestring_col][0][int(col_long)], float): 
                pass # long values are floats
            else:
                print("Error: Latitude column data type is not float")
                sys.exit(1)
            #TODO: turn trajectory csv into point csv
            # x,y = LineString.coords.xy
            # pd.DataFrame({'LAT':x,'LON':y})

        print("All specified columns are valid.")

        standardize_names = {col_id: 'traj_id',
                            col_t: 'timestamp',
                            col_lat: 'latitude',
                            col_long:'longitude',}
        
        df_standardized = df.rename(columns=standardize_names)
        df_standardized = df_standardized[['traj_id', 'timestamp', 'latitude', 'longitude']]
        if file_out is not None:
            csv_name = file_out
        else:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            csv_name = f'{timestr}.csv'

        df_standardized.to_csv(csv_name, index=False)
        return csv_name

def read_file(file_path, filetype='csv', compression=None):
    """Read the file using pandas based on file type."""
    if filetype == 'csv':
        return pd.read_csv(file_path, compression=compression)
    elif filetype == 'xlsx':
        return pd.read_excel(file_path)
    elif filetype == 'json':
        return pd.read_json(file_path)
    elif filetype == "parquet":
        return pd.read_parquet(file_path)
    elif filetype == "feather":
        return pd.read_feather(file_path)
    else:
        raise ValueError(f"Unsupported file type: {filetype} for file \n\t{file_path}")

def main():
    parser = argparse.ArgumentParser(description="Process a file and validate columns.")

    parser.add_argument(
        '--filetype',
        choices=['csv', 'json', 'xlsx', 'parquet', 'feather'],
        required=True,
        help='Type (csv, json, etc.) of file'
    )

    parser.add_argument(
        '--compression',
        type=str,
        default=None,
        help='Type of compression for CSV file'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input file'
    )

    parser.add_argument(
        '--id',
        type=str,
        required=True,
        help='Name of the column containing trajectory ID'
    )

    parser.add_argument(
        '--t',
        type=str,
        required=True,
        help='Name of the column containing timestamp'
    )

    parser.add_argument(
        '--geometry',
        type=str,
        choices=['point', 'linestring'],
        required=True,
        help='"point" if lat/long points are stored in separate columns, one point per row; "linestring" if lat/long points are stored in one column, all points in a row'
    )

    parser.add_argument(
        '--linestring',
        type=str,
        help='Name of the column containing the linestring of locations'
    )

    parser.add_argument(
        '--lat',
        type=str,
        required=True,
        help='Name of the column containing the latitude values or 0/1 value if geometry == linestring, denoting the tuple location of the latitude value'
    )

    parser.add_argument(
        '--long',
        type=str,
        required=True,
        help='Name of the column containing the longitude values or 0/1 value if geometry == linestring, denoting the tuple location of the longitude value'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Optional output file'
    )

    args = parser.parse_args()

    try:
        print("\n############################ START ################################")
        start_time = time.time()
        df = read_file(args.input, args.filetype, args.compression)
        read_time = time.time()
        temp_file_name = validate_columns(df=df, df_columns=df.columns.tolist(), user_columns=[args.id, args.t, args.lat, args.long], geom=args.geometry, file_out=args.output)
        val_col_time = time.time()
        check_file_saved(temp_file_name)
        print(f"FILE UPLOAD TIMES\n\tread in file: {read_time - start_time}\n\tvalidate cols: {val_col_time - read_time}")
    except Exception as e:
        print(f"Failed to process the file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
