#!/bin/bash

################################################################################
# Snowflake ETL Pipeline Orchestrator
#
# This script orchestrates the complete Doppelganger pipeline using Snowflake:
# 1. Sources snow CLI environment
# 2. Sets Snowflake environment variables
# 3. Runs ETL preprocessing in Snowflake (stored procedure)
# 4. Downloads preprocessed data from Snowflake to HPC
# 5. Runs synthetic data generation locally
# 6. Runs audit and privacy evaluation locally
# 7. Uploads synthetic data back to Snowflake
#
# Usage:
#   ./run_snowflake_pipeline.sh <config_name> [options]
#
# Arguments:
#   config_name: Name of the configuration (e.g., 'amc_hf_complete', 'amc_hf_essential')
#
# Options:
#   --snow-cli-path <path>        Path to snow CLI bin/activate (default: auto-detect)
#   --snowflake-home <path>       Path to Snowflake home directory
#   --key-passphrase <pass>       Private key passphrase (or set PRIVATE_KEY_PASSPHRASE env var)
#   --work-dir <path>             Working directory for intermediate files (default: /scratch/sbenson)
#   --gen-config <path>           Path to generation config (default: config/synth.json)
#   --skip-etl                    Skip ETL preprocessing step (use existing tables)
#   --skip-synthesis              Skip synthesis step
#   --skip-audit                  Skip audit step
#   --skip-privacy                Skip privacy evaluation step
#   --skip-upload                 Skip uploading results to Snowflake
#   --output-table <name>         Name for output table in Snowflake (default: <config_name>_SYNTHETIC_DATA)
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SNOW_CLI_PATH="${SNOW_CLI_PATH:-}"
SNOWFLAKE_HOME="${SNOWFLAKE_HOME:-/home/sandbox/sbenson/snowflake}"
PRIVATE_KEY_PASSPHRASE="${PRIVATE_KEY_PASSPHRASE:-}"
WORK_DIR="${WORK_DIR:-/scratch/sbenson}"
GEN_CONFIG="${PROJECT_ROOT}/config/synth.json"
SKIP_ETL=false
SKIP_SYNTHESIS=false
SKIP_AUDIT=false
SKIP_PRIVACY=false
SKIP_UPLOAD=false
OUTPUT_TABLE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 <config_name> [options]

Arguments:
    config_name                   Name of the configuration (e.g., 'amc_hf_complete', 'amc_hf_essential')

Options:
    --snow-cli-path <path>        Path to snow CLI bin/activate (default: auto-detect)
    --snowflake-home <path>       Path to Snowflake home directory (default: /home/sandbox/sbenson/snowflake)
    --key-passphrase <pass>       Private key passphrase (or set PRIVATE_KEY_PASSPHRASE env var)
    --work-dir <path>             Working directory (default: /scratch/sbenson)
    --gen-config <path>           Generation config path (default: config/synth.json)
    --skip-etl                    Skip ETL preprocessing step
    --skip-synthesis              Skip synthesis step
    --skip-audit                  Skip audit step
    --skip-privacy                Skip privacy evaluation step
    --skip-upload                 Skip uploading results to Snowflake
    --output-table <name>         Output table name (default: <config_name>_SYNTHETIC_DATA)
    -h, --help                    Show this help message

Example:
    $0 amc_hf_complete --work-dir /scratch/sbenson

EOF
    exit 1
}

# Parse arguments
if [ $# -eq 0 ]; then
    log_error "No configuration name provided"
    usage
fi

CONFIG_NAME="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --snow-cli-path)
            SNOW_CLI_PATH="$2"
            shift 2
            ;;
        --snowflake-home)
            SNOWFLAKE_HOME="$2"
            shift 2
            ;;
        --key-passphrase)
            PRIVATE_KEY_PASSPHRASE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --gen-config)
            GEN_CONFIG="$2"
            shift 2
            ;;
        --skip-etl)
            SKIP_ETL=true
            shift
            ;;
        --skip-synthesis)
            SKIP_SYNTHESIS=true
            shift
            ;;
        --skip-audit)
            SKIP_AUDIT=true
            shift
            ;;
        --skip-privacy)
            SKIP_PRIVACY=true
            shift
            ;;
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        --output-table)
            OUTPUT_TABLE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Set default output table name if not provided
if [ -z "$OUTPUT_TABLE" ]; then
    OUTPUT_TABLE="${CONFIG_NAME}_SYNTHETIC_DATA"
fi

# Auto-detect snow CLI if not provided
if [ -z "$SNOW_CLI_PATH" ]; then
    log_info "Auto-detecting snow CLI installation..."
    
    # Common locations to check
    POTENTIAL_PATHS=(
        "$HOME/.local/bin/snow"
        "$HOME/bin/snow"
        "/usr/local/bin/snow"
        "$(which snow 2>/dev/null || echo '')"
    )
    
    for path in "${POTENTIAL_PATHS[@]}"; do
        if [ -f "$path" ]; then
            SNOW_CLI_PATH="$path"
            log_success "Found snow CLI at: $SNOW_CLI_PATH"
            break
        fi
    done
    
    if [ -z "$SNOW_CLI_PATH" ]; then
        log_error "Could not find snow CLI. Please install it or specify --snow-cli-path"
        exit 1
    fi
fi

# Verify snow CLI is accessible
if ! command -v snow &> /dev/null; then
    log_error "snow CLI is not in PATH. Please ensure it's properly installed."
    exit 1
fi

# Create work directory
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

log_info "========================================="
log_info "Snowflake ETL Pipeline Orchestrator"
log_info "========================================="
log_info "Configuration: $CONFIG_NAME"
log_info "Working directory: $WORK_DIR"
log_info "Snowflake home: $SNOWFLAKE_HOME"
log_info "Output table: $OUTPUT_TABLE"
log_info "========================================="

# Export Snowflake environment variables
export SNOWFLAKE_HOME="$SNOWFLAKE_HOME"
if [ -n "$PRIVATE_KEY_PASSPHRASE" ]; then
    export SNOWFLAKE_PRIVATE_KEY_PASSPHRASE="$PRIVATE_KEY_PASSPHRASE"
    log_info "Private key passphrase configured"
fi

################################################################################
# Step 1: Upload Configuration to Snowflake
################################################################################
log_info ""
log_info "========================================="
log_info "Step 1: Uploading configuration to Snowflake"
log_info "========================================="

# Determine the config file path
CONFIG_FILE="${PROJECT_ROOT}/config/${CONFIG_NAME}.json"

# Check if it's a full path or just a name
if [ ! -f "$CONFIG_FILE" ]; then
    # Try checking if CONFIG_NAME is already a path
    if [ -f "$CONFIG_NAME" ]; then
        CONFIG_FILE="$CONFIG_NAME"
        # Extract just the name for the table
        CONFIG_NAME=$(basename "$CONFIG_NAME" .json)
    else
        log_error "Configuration file not found: $CONFIG_FILE"
        log_info "Please provide either:"
        log_info "  - A config name (e.g., 'etl_amc_hpc_snowflake_equiv') to look in config/ directory"
        log_info "  - A full path to a JSON config file"
        exit 1
    fi
fi

log_info "Using configuration file: $CONFIG_FILE"

# First, ensure the stored procedure and config table are created
log_info "Creating/updating stored procedure and config table..."
snow sql -f "${SCRIPT_DIR}/snowflake_etl_procedure.sql"

# Read the JSON config
log_info "Reading configuration from file..."
CONFIG_JSON=$(cat "$CONFIG_FILE" | jq -c '.')

if [ -z "$CONFIG_JSON" ] || [ "$CONFIG_JSON" = "null" ]; then
    log_error "Failed to read or parse JSON configuration file"
    exit 1
fi

# Get description from config if available
CONFIG_DESC=$(echo "$CONFIG_JSON" | jq -r '.description // "No description"')

log_info "Uploading configuration '$CONFIG_NAME' to ETL_CONFIGS table..."

# Create a temporary file with config metadata for safer upload
CONFIG_STAGE_FILE="${WORK_DIR}/config_${CONFIG_NAME}.json"
cat > "$CONFIG_STAGE_FILE" << JSONEOF
{
  "config_name": "${CONFIG_NAME}",
  "config_json": ${CONFIG_JSON},
  "description": "${CONFIG_DESC}"
}
JSONEOF

# Upload the config file to Snowflake stage
log_info "Staging configuration file..."
snow sql -q "PUT file://${CONFIG_STAGE_FILE} @~/CONFIG_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;" > /dev/null

# Load from stage into table (handles nested JSON safely)
snow sql -q "
MERGE INTO ETL_CONFIGS AS target
USING (
    SELECT 
        \$1:config_name::STRING AS config_name,
        \$1:config_json::VARIANT AS config_json,
        \$1:description::STRING AS description
    FROM @~/CONFIG_STAGE/config_${CONFIG_NAME}.json
    (FILE_FORMAT => (TYPE = JSON))
) AS source
ON target.config_name = source.config_name
WHEN MATCHED THEN
    UPDATE SET 
        config_json = source.config_json,
        description = source.description,
        updated_date = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
    INSERT (config_name, config_json, description)
    VALUES (source.config_name, source.config_json, source.description);
"

# Clean up staged file
rm -f "$CONFIG_STAGE_FILE"

if [ $? -eq 0 ]; then
    log_success "Configuration uploaded successfully"
else
    log_error "Failed to upload configuration"
    exit 1
fi

################################################################################
# Step 2: Run ETL Preprocessing in Snowflake (Stored Procedure)
################################################################################
if [ "$SKIP_ETL" = false ]; then
    log_info ""
    log_info "========================================="
    log_info "Step 2: Running ETL preprocessing in Snowflake"
    log_info "========================================="
    
    # Call the stored procedure
    log_info "Calling ETL stored procedure with configuration: $CONFIG_NAME"
    ETL_RESULT=$(snow sql -q "CALL dpplgngr_etl_pipeline('${CONFIG_NAME}');" --format json | jq -r '.[0]."DPPLGNGR_ETL_PIPELINE(\047${CONFIG_NAME}\047)"')
    
    if [[ "$ETL_RESULT" == ERROR:* ]]; then
        log_error "ETL preprocessing failed:"
        echo "$ETL_RESULT"
        exit 1
    else
        log_success "ETL preprocessing completed successfully"
        echo "$ETL_RESULT"
    fi
else
    log_warning "Skipping ETL preprocessing step"
fi

################################################################################
# Step 3: Download Preprocessed Data from Snowflake
################################################################################
log_info ""
log_info "========================================="
log_info "Step 3: Downloading preprocessed data"
log_info "========================================="

# Determine the table name based on configuration
PREPROCESSED_TABLE="${CONFIG_NAME}_preprocessed_imputed"
log_info "Downloading from table: $PREPROCESSED_TABLE"

# Download as CSV first, then convert to Parquet
log_info "Downloading data as CSV..."
snow sql -q "SELECT * FROM ${PREPROCESSED_TABLE}" --format csv > "${WORK_DIR}/preprocessed_data.csv"

log_info "Converting CSV to Parquet..."
python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv('${WORK_DIR}/preprocessed_data.csv')
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    df.to_parquet('${WORK_DIR}/preprocessed_data.parquet', index=False)
    print("Successfully converted to Parquet format")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    log_success "Data downloaded and converted successfully"
    # Clean up CSV to save space
    rm -f "${WORK_DIR}/preprocessed_data.csv"
else
    log_error "Failed to download or convert data"
    exit 1
fi

################################################################################
# Step 4: Run Synthetic Data Generation
################################################################################
if [ "$SKIP_SYNTHESIS" = false ]; then
    log_info ""
    log_info "========================================="
    log_info "Step 4: Running synthetic data generation"
    log_info "========================================="    # Create temporary ETL config for synthesis step
    ETL_CONFIG_TEMP="${WORK_DIR}/etl_config_temp.json"
    cat > "$ETL_CONFIG_TEMP" << EOF
{
    "name": "${CONFIG_NAME}",
    "preprocessing": "${WORK_DIR}",
    "preprocessed_file": "${WORK_DIR}/preprocessed_data.parquet"
}
EOF
    
    log_info "Running synthesis with config: $GEN_CONFIG"
    python3 "${SCRIPT_DIR}/run_synthesis_local.py" \
        --etl-config "$ETL_CONFIG_TEMP" \
        --gen-config "$GEN_CONFIG" \
        --output-dir "$WORK_DIR"
    
    if [ $? -eq 0 ]; then
        log_success "Synthetic data generation completed"
    else
        log_error "Synthetic data generation failed"
        exit 1
    fi
else
    log_warning "Skipping synthesis step"
fi

################################################################################
# Step 5: Run Audit Evaluation
################################################################################
if [ "$SKIP_AUDIT" = false ]; then
    log_info ""
    log_info "========================================="
    log_info "Step 5: Running audit evaluation"
    log_info "========================================="
    
    python3 "${SCRIPT_DIR}/run_synthesis_local.py" \
        --etl-config "$ETL_CONFIG_TEMP" \
        --gen-config "$GEN_CONFIG" \
        --output-dir "$WORK_DIR" \
        --audit-only
    
    if [ $? -eq 0 ]; then
        log_success "Audit evaluation completed"
    else
        log_warning "Audit evaluation failed (continuing anyway)"
    fi
else
    log_warning "Skipping audit step"
fi

################################################################################
# Step 6: Run Privacy Evaluation
################################################################################
if [ "$SKIP_PRIVACY" = false ]; then
    log_info ""
    log_info "========================================="
    log_info "Step 6: Running privacy evaluation"
    log_info "========================================="
    
    python3 "${SCRIPT_DIR}/run_synthesis_local.py" \
        --etl-config "$ETL_CONFIG_TEMP" \
        --gen-config "$GEN_CONFIG" \
        --output-dir "$WORK_DIR" \
        --privacy-only
    
    if [ $? -eq 0 ]; then
        log_success "Privacy evaluation completed"
    else
        log_warning "Privacy evaluation failed (continuing anyway)"
    fi
else
    log_warning "Skipping privacy step"
fi

################################################################################
# Step 7: Upload Synthetic Data to Snowflake
################################################################################
if [ "$SKIP_UPLOAD" = false ]; then
    log_info ""
    log_info "========================================="
    log_info "Step 7: Uploading synthetic data to Snowflake"
    log_info "========================================="
    
    SYNTHETIC_FILE="${WORK_DIR}/synthetic_data.parquet"
    
    if [ ! -f "$SYNTHETIC_FILE" ]; then
        log_error "Synthetic data file not found: $SYNTHETIC_FILE"
        exit 1
    fi
    
    log_info "Uploading file to Snowflake stage..."
    snow sql -q "PUT file://${SYNTHETIC_FILE} @~/SYNTHETIC_DATA_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
    
    if [ $? -ne 0 ]; then
        log_error "Failed to upload file to Snowflake stage"
        exit 1
    fi
    
    log_info "Creating table from staged file..."
    snow sql -q "
CREATE OR REPLACE TABLE ${OUTPUT_TABLE} 
USING TEMPLATE (
    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
    FROM TABLE(
        INFER_SCHEMA(
            LOCATION=>'@~/SYNTHETIC_DATA_STAGE/',
            FILE_FORMAT=>'PARQUET_FORMAT'
        )
    )
);

COPY INTO ${OUTPUT_TABLE}
FROM @~/SYNTHETIC_DATA_STAGE/
FILE_FORMAT = (TYPE = PARQUET)
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
"
    
    if [ $? -eq 0 ]; then
        log_success "Synthetic data uploaded to table: $OUTPUT_TABLE"
        
        # Get row count
        ROW_COUNT=$(snow sql -q "SELECT COUNT(*) as CNT FROM ${OUTPUT_TABLE};" --format json | jq -r '.[0].CNT')
        log_info "Table contains $ROW_COUNT rows"
    else
        log_error "Failed to create table from staged file"
        exit 1
    fi
else
    log_warning "Skipping upload step"
fi

################################################################################
# Step 7: Summary
################################################################################
log_info ""
log_info "========================================="
log_info "Pipeline Execution Summary"
log_info "========================================="
log_success "All steps completed successfully!"
log_info ""
log_info "Output files:"
log_info "  - Preprocessed data: ${WORK_DIR}/preprocessed_data.parquet"
log_info "  - Synthetic data: ${WORK_DIR}/synthetic_data.parquet"
if [ "$SKIP_AUDIT" = false ]; then
    log_info "  - Audit results: ${WORK_DIR}/audit_results.json"
fi
if [ "$SKIP_PRIVACY" = false ]; then
    log_info "  - Privacy results: ${WORK_DIR}/privacy_results.json"
fi
log_info ""
log_info "Snowflake tables:"
log_info "  - Preprocessed: ${CONFIG_NAME}_preprocessed_imputed"
if [ "$SKIP_UPLOAD" = false ]; then
    log_info "  - Synthetic: ${OUTPUT_TABLE}"
fi
log_info ""
log_info "========================================="
