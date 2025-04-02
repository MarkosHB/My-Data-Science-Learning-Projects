import pandas as pd

#####################
# DataFrame Lifecycle
#####################

# Create a df Dataframe with determined information.
df = pd.DataFrame(
    data=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]], 
    columns=["A", "B", "C"], 
    index=["x","y","z","w"]
)

# Load data from a file.
df = pd.read_csv("data.csv")  # Read a CSV file into a DataFrame
df = pd.read_excel("data.xlsx")  # Read an Excel file into a DataFrame

# Save as a file.
df.to_csv("data.csv")  # Save as a CSV file
df.to_excel("data.xlsx")  # Sava aa EXCEL file 


########################################
# Obtain information about the DataFrame
########################################

df.head(5)  # Display the first 5 rows of the DataFrame.
df.tail(5)  # Display the last 5 rows of the DataFrame.
df.sample(3)  # Randomly select 3 rows from the DataFrame.
df.info()  # Display a concise summary of the DataFrame.
df.describe()  # Generate descriptive statistics of the DataFrame.
df.dtypes  # Get the data types of the DataFrame.
df.columns  # Get the column names of the DataFrame.
df.index.tolist() # Get the index of the DataFrame as a list.

df2 = df.copy()  # Create a copy of the DataFrame.
df.merge(df2, on="key")  # Merge two DataFrames on a key column.
df.concat([df2, df])  # Concatenate two DataFrames along rows.

df.shape[0]  # Get the shape of the DataFrame (rows). 
df.shape[1]  # Get the shape of the DataFrame (columns).
df.ndim  # Get the number of dimensions of the DataFrame.
df.size  # Get number of elements in the DataFrame.

df.quantiles()
df.max()
df.min()
df.sum()
df.mean()
df.median()
df.std()
df.var()
df.corr()
df.cov()

#####################
# DataFrame Selection
#####################

df["A"] # Select a column // df.A
df.filter(items=["A", "B"])  # df["A", "B"]
df["A"].unique() # Get unique values in a column.
df.index = df["A"] # Set index to a column.
df.nunique()  # Get the number of unique values in each column.
df["A"].value_counts()  # Count unique values in a column.

df.loc[["x", "y"], "B"] # df.loc[Rows_index, 'Columns_name']
df.iloc[[1, 2], 2]  # df.iloc[Rows_index_pos, Columns_name_pos]
df.at["x", "B"]  # df.at[Row_index, 'Column_name'] // Only one value.
df.iat[1, 2]  # df.iat[Row_index_pos, Column_name_pos] // Only one value.

df[df["A"] > 5]["B"]  # Select rows based on a condition and display column. 
df.loc[df["A"] > 5, "B"]  # Select rows based on a condition and display column.
df[(df["A"] > 5) & df["C"] < 10] # Select rows based on multiple contitions.

df.query("A > 5 and C < 10") # Select rows by quering the DataFrame.
df.sort_values(["A", "B"], ascending=[0,1]) # Sort DataFrame by columns.
df.groupby(["A", "B"])["C"].sum()  # Group by columns and sum another column.
df.groupby(["A"]).agg({"B": "sum", "C": "mean"})  # Group by a column and sum another column.
df.pivot(columns="A", index="B", values="C")  # Pivot the DataFrame.
df.iterrows()  # for index, row in df.iterrows(): 


########################
# DataFrame manipulation
########################

df["new_row"] = "example" # Add a new column.
df.pop("A")  # Remove a column and return it.
df.drop("A")  # Remove a column without returning it.
df.rename(columns={"A": "new_A"})  # Rename a column.
df["A"].to_datetime(format="%Y-%m-%d")  # Convert a column to datetime.
df["A"].dt.year  # Extract the year (month, day) from a datetime column.

df.isna().sum()  # Count NaN values in each column.
df.dropna()  # Remove rows with NaN values.
df.drop_duplicates() # Remove duplicate rows.
df["A"].interpolate()  # Interpolate missing values.
df.ffill()  # Fullfill NaN values with last valid entry.
df.fillna(value={{"A": 0}})  # Fill NaN values with a specific value.

df["A"].apply(lambda x: x + 1)  # Apply a function to each element.
df["A"].map(lambda x: x + 1)  # Map a function to each element.
