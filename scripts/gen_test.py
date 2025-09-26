import random
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

# Generate file 1: 1000 records with id, age, gender
def generate_file1():
    data = []
    
    for i in range(1000):
        id_val = i + 1
        age = random.randint(20, 60)
        gender = random.randint(0, 1)
        data.append([id_val, age, gender])
    
    df = pd.DataFrame(data, columns=['id', 'age', 'gender'])
    df.to_parquet('data_file1.parquet', index=False)

# Generate file 2: 3000 records with id, measurement, measurement_type, measurement_date
def generate_file2():
    data = []
    
    # Date range for random dates
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days
    
    for i in range(3000):
        id_val = random.randint(1, 1000)  # Same range as file 1
        measurement = round(random.uniform(1.0, 10.0), 2)
        measurement_type = random.randint(1, 3)
        
        # Generate random date
        random_days = random.randint(0, date_range)
        random_date = start_date + timedelta(days=random_days)
        date_str = random_date.strftime('%Y-%m-%d')
        
        data.append([id_val, measurement, measurement_type, date_str])
    
    df = pd.DataFrame(data, columns=['id', 'measurement', 'measurement_type', 'measurement_date'])
    df.to_parquet('data_file2.parquet', index=False)

if __name__ == "__main__":
    generate_file1()
    generate_file2()
    print("Generated data_file1.parquet (1000 records) and data_file2.parquet (3000 records)")