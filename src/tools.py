def clean_sqft_basement(sqft):
    if sqft =='?':
        return 0.0
    return float(sqft)

def print_column_info(column):
    print(f"Unique values:\n {column.unique()}")
    print(f"Value_counts:\n {column.value_counts()}")
    print(f"Number of Null values:\n {column.isnull().sum()}")
