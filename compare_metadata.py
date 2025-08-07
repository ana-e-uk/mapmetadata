# %%
import pandas as pd

df = pd.read_csv('/home/spatialuser/uribe/mapmetadata/jakarta2_inf_metadata.csv')


# %%
df.head()

# %%
def resolve_oneway(group):
    if False in group.values:
        return False
    elif True in group.values:
        return True
    else:
        return False
# Group by 'osmid' and apply logic to 'oneway'
result = df.groupby('osmid')['Oneway'].apply(resolve_oneway).reset_index()


# %%
result.head()

#%%
result.shape

# %%

import psycopg2
from psycopg2.extras import execute_values

# Connect to DB
conn = psycopg2.connect(
    dbname="gis",
    user="gis",
    password="gis",
    host="cs-u-spatial-406.cs.umn.edu",  # or wherever your DB is hosted
    port=5432
)
cur = conn.cursor()

# Prepare data as list of tuples
data = list(result.itertuples(index=False, name=None))

# Perform UPSERT
query = """
UPDATE roads
SET inf_oneway_direction = data.inf_oneway_direction
FROM (VALUES %s) AS data(id, inf_oneway_direction)
WHERE roads.id = data.id;
"""

# Efficient batch insert
execute_values(cur, query, data)

conn.commit()

query2 = """
SELECT 
    COUNT(CASE 
            WHEN osm_oneway_direction = inf_oneway_direction THEN 1 
         END) * 1.0 
    / COUNT(*) AS accuracy
FROM roads
WHERE inf_oneway_direction IS NOT NULL;
"""

cur.execute(query2)
result = cur.fetchone()
print("Accuracy:", result[0])

cur.close()
conn.close()
# %%
def resolve_oneway_osm(group):
    if False in group.values:
        return False
    elif True in group.values:
        return True
    else:
        return None

# agg_result = df.groupby('osmid').agg({
#     'Oneway': resolve_oneway,
#     'OSM_oneway': resolve_oneway_osm
# }).reset_index()


# # %%
# agg_result

# # %%
# accuracy = (df['Oneway'] == df['OSM_oneway']).mean()
# print(f"Accuracy: {accuracy:.4f}")
# # %%