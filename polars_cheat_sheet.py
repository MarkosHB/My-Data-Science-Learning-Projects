import polars as pl

#####################
# DataFrame Lifecycle
#####################

# Create a DataFrame with determined information.
df = pl.DataFrame(
    data={"A": [1, 4, 7, 10], "B": [2, 5, 8, 11], "C": [3, 6, 9, 12]},
    schema=["A", "B", "C"]
)

# Load data from a file.
df = pl.read_csv("data.csv")  # Read a CSV file into a DataFrame
df = pl.read_excel("data.xlsx")  # Read an Excel file into a DataFrame

# Save as a file.
df.write_csv("data.csv")  # Save as a CSV file
df.write_excel("data.xlsx")  # Save as an Excel file


########################################
# Obtain information about the DataFrame
########################################

df.head(5)  # Display the first 5 rows of the DataFrame.
df.tail(5)  # Display the last 5 rows of the DataFrame.
df.sample(n=3)  # Randomly select 3 rows from the DataFrame.
df.describe()  # Generate descriptive statistics of the DataFrame.
df.schema  # Get the schema (data types) of the DataFrame.
df.columns  # Get the column names of the DataFrame.
df.height  # Get the number of rows in the DataFrame.
df.width  # Get the number of columns in the DataFrame.

df2 = df.clone()  # Create a copy of the DataFrame.
df.join(df2, on="key")  # Join two DataFrames on a key column.
pl.concat([df2, df])  # Concatenate two DataFrames along rows.

df.shape[0]  # Get the shape of the DataFrame (rows).
df.shape[1]  # Get the shape of the DataFrame (columns).

df.select(pl.col("A").quantile(0.5))  # Quantiles
df.select(pl.col("A").max())  # Max
df.select(pl.col("A").min())  # Min
df.select(pl.col("A").sum())  # Sum
df.select(pl.col("A").mean())  # Mean
df.select(pl.col("A").median())  # Median
df.select(pl.col("A").std())  # Standard deviation
df.select(pl.col("A").var())  # Variance
df.corr()  # Correlation
df.select(pl.cov("A", "B"))  # Covariance


#####################
# DataFrame Selection
#####################

df.select("A")  # Select a column
df.select(["A", "B"])  # Select multiple columns
df.select(pl.col("A").unique())  # Get unique values in a column.
df.with_columns(pl.col("A").alias("index"))  # Set index to a column.
df.n_unique()  # Get the number of unique values in each column.
df.select(pl.col("A").value_counts())  # Count unique values in a column.

df.filter(pl.col("A") > 5).select("B")  # Select rows based on a condition and display column.
df.filter((pl.col("A") > 5) & (pl.col("C") < 10))  # Select rows based on multiple conditions.

df.sort(["A", "B"], descending=[True, False])  # Sort DataFrame by columns.
df.group_by(["A", "B"]).agg(pl.col("C").sum())  # Group by columns and sum another column.
df.group_by("A").agg([pl.col("B").sum(), pl.col("C").mean()])  # Group by a column and aggregate.
df.pivot(values="C", on="A", index="B")  # Pivot the DataFrame.


########################
# DataFrame manipulation
########################

df.with_columns(pl.lit("example").alias("new_row"))  # Add a new column.
df.drop("A")  # Remove a column.
df.rename({"A": "new_A"})  # Rename a column.
df.with_columns(pl.col("A").str.strptime(pl.Date, format="%Y-%m-%d"))  # Convert a column to datetime.
df.with_columns(pl.col("A").dt.year())  # Extract the year from a datetime column.

df.select(pl.all().is_null().sum())  # Count NaN values in each column.
df.drop_nulls()  # Remove rows with NaN values.
df.unique()  # Remove duplicate rows.
df.with_columns(pl.col("A").interpolate())  # Interpolate missing values.
df.fill_null({"A": 0})  # Fill NaN values with a specific value.

df.with_columns( 
    pl.col("A")
    .map_elements(lambda x: x * 2, return_dtype=pl.Int64)
    .alias("A_times_2"),
) # Apply a function to each element.