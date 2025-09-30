#!/bin/bash

# Doppelganger Apptainer Build Script (using uv for fast Python package management)
# Author: SB
# Date: 2025-09-28
# Updated: 2025-01-22 - Switched from conda to uv for faster builds

set -e

# Configuration
CONTAINER_NAME="doppelganger.sif"
DEF_FILE="doppelganger.def"
BUILD_DIR="build"  # Default value, will be updated in argument parsing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Doppelganger Container Build  ${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_dependencies() {
    print_step "Checking dependencies..."
    
    if ! command -v apptainer &> /dev/null; then
        print_error "Apptainer is not installed or not in PATH"
        echo "Please install Apptainer: https://apptainer.org/docs/admin/main/installation.html"
        exit 1
    fi
    
    if [ ! -f "$DEF_FILE" ]; then
        print_error "Definition file $DEF_FILE not found"
        exit 1
    fi
    
    if [ ! -d "dpplgngr" ]; then
        print_error "dpplgngr package directory not found"
        echo "Please run this script from the Doppelganger root directory"
        exit 1
    fi
    
    print_success "All dependencies found"
}

create_build_dir() {
    print_step "Creating build directory..."
    mkdir -p "$BUILD_DIR"
    print_success "Build directory created: $BUILD_DIR"
}

build_container() {
    print_step "Building Apptainer container..."
    print_info "This may take 10-20 minutes depending on your internet connection..."
    
    # Build with fakeroot (recommended) or sudo
    if apptainer build --help | grep -q "fakeroot"; then
        print_info "Building with fakeroot..."
        apptainer build --fakeroot "$BUILD_DIR/$CONTAINER_NAME" "$DEF_FILE"
    else
        print_info "Building with sudo (fakeroot not available)..."
        sudo apptainer build "$BUILD_DIR/$CONTAINER_NAME" "$DEF_FILE"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Container built successfully: $BUILD_DIR/$CONTAINER_NAME"
    else
        print_error "Container build failed"
        exit 1
    fi
}

test_container() {
    print_step "Testing container..."
    
    # Test basic functionality
    print_info "Testing Python environment..."
    apptainer exec "$BUILD_DIR/$CONTAINER_NAME" python --version
    
    print_info "Testing package imports..."
    apptainer exec "$BUILD_DIR/$CONTAINER_NAME" python -c "
import sys
print(f'Python: {sys.version}')
import numpy, pandas, luigi, sdv
print('Core packages: OK')
from dpplgngr.train.sdv import SDVGen
from dpplgngr.scores.audit import SyntheticDataAudit
print('Doppelganger packages: OK')
"
    
    if [ $? -eq 0 ]; then
        print_success "Container tests passed"
    else
        print_error "Container tests failed"
        exit 1
    fi
}

show_usage_examples() {
    echo ""
    echo -e "${GREEN}Container built successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Usage Examples:${NC}"
    echo ""
    echo "1. Interactive shell:"
    echo "   apptainer shell $BUILD_DIR/$CONTAINER_NAME"
    echo ""
    echo "2. Run ETL pipeline:"
    echo "   apptainer run $BUILD_DIR/$CONTAINER_NAME etl"
    echo ""
    echo "3. Run synthetic data generation:"
    echo "   apptainer run $BUILD_DIR/$CONTAINER_NAME synth"
    echo ""
    echo "4. Run data audit:"
    echo "   apptainer run $BUILD_DIR/$CONTAINER_NAME audit"
    echo ""
    echo "5. Run tests:"
    echo "   apptainer run $BUILD_DIR/$CONTAINER_NAME test"
    echo ""
    echo "6. Execute custom Python script:"
    echo "   apptainer exec $BUILD_DIR/$CONTAINER_NAME python your_script.py"
    echo ""
    echo "7. Bind mount data directories:"
    echo "   apptainer exec -B /path/to/data:/data $BUILD_DIR/$CONTAINER_NAME python script.py"
    echo ""
    echo -e "${BLUE}Container location:${NC} $(pwd)/$BUILD_DIR/$CONTAINER_NAME"
    echo -e "${BLUE}Container size:${NC} $(du -sh "$BUILD_DIR/$CONTAINER_NAME" | cut -f1)"
}

# Main execution
main() {
    print_header
    
    # Parse command line arguments
    SKIP_TESTS=false
    FORCE_REBUILD=false
    
    # First, check if the first argument is a directory path (not starting with --)
    if [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]] && [[ "$1" != "-h" ]]; then
        BUILD_DIR="$1"
        shift
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --force)
                FORCE_REBUILD=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [BUILD_DIR] [OPTIONS]"
                echo ""
                echo "Arguments:"
                echo "  BUILD_DIR       Directory to store the built container (default: build)"
                echo ""
                echo "Options:"
                echo "  --skip-tests    Skip container testing after build"
                echo "  --force         Force rebuild even if container exists"
                echo "  --help, -h      Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                                    # Build in ./build/"
                echo "  $0 /tmp/containers                    # Build in /tmp/containers/"
                echo "  $0 /scratch/build --force             # Force rebuild in /scratch/build/"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check if container already exists
    if [ -f "$BUILD_DIR/$CONTAINER_NAME" ] && [ "$FORCE_REBUILD" = false ]; then
        print_info "Container already exists: $BUILD_DIR/$CONTAINER_NAME"
        echo "Use --force to rebuild"
        show_usage_examples
        exit 0
    fi
    
    check_dependencies
    create_build_dir
    build_container
    
    if [ "$SKIP_TESTS" = false ]; then
        test_container
    fi
    
    show_usage_examples
}

# Run main function
main "$@"
