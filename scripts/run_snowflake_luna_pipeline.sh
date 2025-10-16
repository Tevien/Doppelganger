#!/bin/bash
#SBATCH --job-name=snowflake_pipeline
#SBATCH --output=pipeline_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Make sure we find the right config
export SNOWFLAKE_HOME="/home/sandbox/sbenson/snowflake"
# Get arguments passed to the script
PRIVATE_KEY_PASSPHRASE=$1
# Use the arguments in your script
echo "Using private key passphrase: ${PRIVATE_KEY_PASSPHRASE:0:4}****"  # Only show first 4 chars for security
python your_pipeline.py --private-key "$PRIVATE_KEY_PASSPHRASE"

# Get python env
source /home/sandbox/sbenson/Doppelganger/dpplenv/bin/activate

# Go to working directory
cd /scratch/sbenson

# Download as CSV using Snowflake CLI
snow sql -q "SELECT * FROM AMC_PROCESS_TUPLEPROCESSED" --format csv > data.csv

# Convert to Parquet using Python
python -c "
import pandas as pd
df = pd.read_csv('data.csv')
df.to_parquet('data.parquet', index=False)
"

# Run the synthesis pipeline
python /home/sandbox/sbenson/Doppelganger/scripts/run_luna.py

# Upload local file to Snowflake stage
snow sql -q "PUT file://./synthetic_data.parquet @~/STRD/ AUTO_COMPRESS=FALSE;"

# Upload the results back to Snowflake
snow sql -q "
CREATE OR REPLACE TABLE AMC_SYNTHETIC_DATA AS
SELECT * FROM @~/STRD/synthetic_data.parquet
FILE_FORMAT = (TYPE = PARQUET);
"