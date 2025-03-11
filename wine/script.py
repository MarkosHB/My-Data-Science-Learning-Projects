# %%
import pandas as pd
from sklearn import datasets
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf

# Init spark session.
spark = SparkSession.builder \
    .appName("Wine dataset with Spark") \
    .getOrCreate()

# %%
ds = datasets.load_wine(as_frame=True)
# Parse dataset into spark.
df = spark.createDataFrame(ds.frame)

# Check total number of rows and columns in the dataset.
print(f"Total number of samples is '{df.count()}' and there are '{len(df.columns)}' columns.", end="\n\n")

# Obtain a brief look of the dataset.
df.show()
df.printSchema()

# (Optional) See dataset description.
# print(f"Dataset contextual information:\n {ds.DESCR}")

# %%
# Clean loaded data (if needed).
print(f"There are {df.drop_duplicates().count()-df.count()} columns duplicated and therefore removed from the dataframe.")
print(f"There were {df.dropna().count()-df.count()} columns empty and therefore removed from the dataframe.")

# %%
# General details of dataframe and its columns.
df.select("*").describe().show(vertical=True)

# %%
# Obtain first column.
print(df.take(1))

# Obtain all ocurrences.
# df.collect()

# %%
# Print a determined column.
df.select(df.alcohol).show()

# %%
# Print table with column condition.
df.filter(df.alcohol > 13).show()

# %%
# Collapse columns from its unique values.
df.groupby("target").avg().show(vertical=True)

# %%
# Define a temporal view from current Spark Session.
df.createOrReplaceTempView("tableA")
spark.sql("SELECT count(*) from tableA").show()

# %%
# Define custom pandas functions.
@pandas_udf("integer")
def add_one(s: pd.Series) -> pd.Series:
    return s + 1

# Make them accesible from Spark.
spark.udf.register("add_one", add_one)
spark.sql("SELECT DISTINCT add_one(target) FROM tableA").show()
