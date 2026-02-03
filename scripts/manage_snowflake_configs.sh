#!/bin/bash

################################################################################
# Snowflake Configuration Manager
#
# Helper script to manage ETL configurations in Snowflake
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

usage() {
    cat << EOF
Snowflake Configuration Manager

Usage: $0 <command> [options]

Commands:
    list                          List all configurations in Snowflake
    show <config_name>            Show details of a specific configuration
    upload <config_file>          Upload a configuration from JSON file
    delete <config_name>          Delete a configuration
    export <config_name> [file]   Export a configuration to JSON file

Options:
    -h, --help                    Show this help message

Examples:
    # List all configs
    $0 list

    # Show a specific config
    $0 show etl_amc_hpc_snowflake_equiv

    # Upload a new config
    $0 upload config/my_config.json

    # Export a config from Snowflake
    $0 export etl_amc_hpc_snowflake_equiv my_backup.json

    # Delete a config
    $0 delete old_config

EOF
    exit 1
}

if [ $# -eq 0 ]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    list)
        log_info "Listing all configurations in Snowflake..."
        snow sql -q "
            SELECT 
                config_name,
                description,
                TO_VARCHAR(created_date, 'YYYY-MM-DD HH24:MI:SS') as created,
                TO_VARCHAR(updated_date, 'YYYY-MM-DD HH24:MI:SS') as updated
            FROM ETL_CONFIGS 
            ORDER BY config_name
        " --format table
        ;;
    
    show)
        if [ $# -eq 0 ]; then
            echo "Error: config_name required"
            echo "Usage: $0 show <config_name>"
            exit 1
        fi
        CONFIG_NAME="$1"
        log_info "Showing configuration: $CONFIG_NAME"
        snow sql -q "
            SELECT 
                config_name,
                description,
                config_json,
                TO_VARCHAR(created_date, 'YYYY-MM-DD HH24:MI:SS') as created,
                TO_VARCHAR(updated_date, 'YYYY-MM-DD HH24:MI:SS') as updated
            FROM ETL_CONFIGS 
            WHERE config_name = '${CONFIG_NAME}'
        " --format json | jq '.'
        ;;
    
    upload)
        if [ $# -eq 0 ]; then
            echo "Error: config_file required"
            echo "Usage: $0 upload <config_file>"
            exit 1
        fi
        CONFIG_FILE="$1"
        
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "Error: File not found: $CONFIG_FILE"
            exit 1
        fi
        
        CONFIG_NAME=$(basename "$CONFIG_FILE" .json)
        log_info "Uploading configuration from: $CONFIG_FILE"
        log_info "Configuration name: $CONFIG_NAME"
        
        CONFIG_JSON=$(cat "$CONFIG_FILE" | jq -c '.')
        CONFIG_DESC=$(echo "$CONFIG_JSON" | jq -r '.description // "No description"')
        
        snow sql -q "
            MERGE INTO ETL_CONFIGS AS target
            USING (SELECT 
                '${CONFIG_NAME}' AS config_name,
                PARSE_JSON('${CONFIG_JSON}') AS config_json,
                '${CONFIG_DESC}' AS description
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
        
        log_success "Configuration uploaded: $CONFIG_NAME"
        ;;
    
    export)
        if [ $# -eq 0 ]; then
            echo "Error: config_name required"
            echo "Usage: $0 export <config_name> [output_file]"
            exit 1
        fi
        CONFIG_NAME="$1"
        OUTPUT_FILE="${2:-${CONFIG_NAME}.json}"
        
        log_info "Exporting configuration: $CONFIG_NAME"
        log_info "Output file: $OUTPUT_FILE"
        
        snow sql -q "
            SELECT config_json
            FROM ETL_CONFIGS 
            WHERE config_name = '${CONFIG_NAME}'
        " --format json | jq -r '.[0].CONFIG_JSON' | jq '.' > "$OUTPUT_FILE"
        
        if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
            log_success "Configuration exported to: $OUTPUT_FILE"
        else
            echo "Error: Failed to export configuration"
            exit 1
        fi
        ;;
    
    delete)
        if [ $# -eq 0 ]; then
            echo "Error: config_name required"
            echo "Usage: $0 delete <config_name>"
            exit 1
        fi
        CONFIG_NAME="$1"
        
        echo -e "${YELLOW}Warning: This will delete configuration '$CONFIG_NAME'${NC}"
        read -p "Are you sure? (yes/no): " CONFIRM
        
        if [ "$CONFIRM" != "yes" ]; then
            echo "Cancelled"
            exit 0
        fi
        
        log_info "Deleting configuration: $CONFIG_NAME"
        snow sql -q "DELETE FROM ETL_CONFIGS WHERE config_name = '${CONFIG_NAME}'"
        
        log_success "Configuration deleted: $CONFIG_NAME"
        ;;
    
    -h|--help)
        usage
        ;;
    
    *)
        echo "Error: Unknown command: $COMMAND"
        usage
        ;;
esac
