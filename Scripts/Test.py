import pandas as pd

def load_data(file_path):
    """
    Load a CSV file and return a DataFrame.
    """
    try:
        # Try to read the file
        data = pd.read_csv(file_path)

        # Check if the DataFrame is empty (i.e., no rows/columns)
        if data.empty:
            print(f"Warning: The file at {file_path} is empty.")
            return None
        else:
            print(f"Data loaded successfully from {file_path}")
            return data
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty or has no data.")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def view_data(data, n=5):
    """
    Display the first `n` rows of the data.
    Default is 5 rows.
    """
    if data is not None:
        print(data.head(n))
    else:
        print("No data to display.")

alert_data=pd.read_csv("/Users/nidhipatel/Desktop/Classes/Winter25/Experiential Learning/HarmReduction-Project/Data/Raw/HealthCanada-SubstanceHarmsData.csv")
view_data(alert_data)