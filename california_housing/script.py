# %%
from sklearn.datasets import fetch_california_housing
from dask.distributed import Client
import dask.dataframe as dd

# Parallel computing.
client = Client()

# %%
ds = fetch_california_housing(as_frame=True)
# Parse dataset into dask.
ddf = dd.from_pandas(ds.frame, npartitions=4)

# Check total number of rows and columns in the dataset.
print(f"Total number of samples is '{ddf.shape[0].compute()}' and there are '{len(ddf.columns)}' columns.", end="\n\n")

# Obtain a brief look of the dataset.
print(ddf.head(), end="\n\n")  # Muestra las primeras filas

# Show data types.
print(ddf.dtypes) 

# (Optional) See dataset description.
# print(f"Dataset contextual information:\n {ds.DESCR}")

# %%
# Obtain general metrics.
print(ddf.describe().compute())

# %%
# Filter column with condition.
print(ddf[ddf['MedHouseVal'] > 3].head())

# %%
# Collapse columns from its unique values.
print(ddf.groupby("AveRooms")["MedHouseVal"].mean().compute())

# %%
# Apply function in a single column (WITHOUT PARALLEL COMPUTING).
print(ddf["Population"].apply(lambda x: x/2, meta=('Population', 'float64')).head())

# %%
# Self-defined function.
def increment_prices(df):
    df['MedHouseVal'] = df['MedHouseVal'] * 1.1  # +10%
    return df

# Apply in all partitions (PARALLEL COMPUTING).
print(ddf.map_partitions(increment_prices).head())


