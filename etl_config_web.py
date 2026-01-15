#!/usr/bin/env python3
"""
ETL Configuration Builder Web GUI

A Flask-based web interface for building ETL JSON/Python configuration files.
Users can select files, choose columns, and configure measurement mappings through a browser.
"""

from flask import Flask, render_template, request, jsonify, send_file, session
import json
import os
from pathlib import Path
import pandas as pd
from werkzeug.utils import secure_filename
import tempfile
import secrets
import subprocess
import platform

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global storage for configurations (in production, use a database)
configurations = {}


def read_file_columns(file_path):
    """Read columns from a data file"""
    try:
        if file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=0)
        else:
            return None, "Unsupported file format"
        
        return list(df.columns), None
    except Exception as e:
        return None, str(e)


def build_json_config(config_data):
    """Build JSON configuration from form data"""
    config = {
        "name": config_data.get("name", "unnamed_process"),
        "description": config_data.get("description", "ETL configuration"),
        "absolute_path": config_data.get("absolute_path", ""),
        "preprocessing": config_data.get("preprocessing", "")
    }
    
    file_configs = config_data.get("file_configs", {})
    
    for filename, file_config in file_configs.items():
        columns = file_config["columns"].copy()
        
        # Handle measurement mapping
        if file_config.get("mapping"):
            mapping_info = file_config["mapping"]
            mapping_dict = mapping_info["mappings"]
            
            if mapping_info["mapping_col"] in columns:
                idx = columns.index(mapping_info["mapping_col"])
                columns[idx] = mapping_dict
        
        # Handle date column
        if file_config.get("date_column"):
            date_col = file_config["date_column"]
            if date_col in columns:
                idx = columns.index(date_col)
                columns[idx] = [date_col]
        
        config[filename] = columns
    
    return config


def build_python_config(config_data):
    """Build Python configuration string from form data"""
    lines = ["# ETL Configuration", ""]
    
    lines.append(f"config = {{")
    lines.append(f'    "name": "{config_data.get("name", "unnamed_process")}",')
    lines.append(f'    "description": "{config_data.get("description", "ETL configuration")}",')
    lines.append(f'    "absolute_path": "{config_data.get("absolute_path", "")}",')
    lines.append(f'    "preprocessing": "{config_data.get("preprocessing", "")}",')
    
    file_configs = config_data.get("file_configs", {})
    
    for filename, file_config in file_configs.items():
        lines.append(f'    "{filename}": [')
        
        columns = file_config["columns"].copy()
        
        for col in columns:
            # Check if this is a mapping column
            if file_config.get("mapping") and col == file_config["mapping"]["mapping_col"]:
                mapping_dict = file_config["mapping"]["mappings"]
                lines.append(f'        {repr(mapping_dict)},')
            # Check if this is a date column
            elif file_config.get("date_column") and col == file_config["date_column"]:
                lines.append(f'        ["{col}"],')
            else:
                lines.append(f'        "{col}",')
        
        lines.append(f'    ],')
    
    lines.append(f'}}')
    
    return '\n'.join(lines)


@app.route('/')
def index():
    """Main page"""
    session_id = session.get('session_id')
    if not session_id:
        session_id = secrets.token_hex(8)
        session['session_id'] = session_id
        configurations[session_id] = {
            "name": "amc_process",
            "description": "Preprocessing of AMC data",
            "absolute_path": "",
            "preprocessing": "",
            "file_configs": {}
        }
    
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    session_id = session.get('session_id')
    if not session_id or session_id not in configurations:
        return jsonify({"error": "No configuration found"}), 404
    
    return jsonify(configurations[session_id])


@app.route('/api/config/basic', methods=['POST'])
def update_basic_config():
    """Update basic configuration"""
    session_id = session.get('session_id')
    if not session_id or session_id not in configurations:
        return jsonify({"error": "No configuration found"}), 404
    
    data = request.json
    config = configurations[session_id]
    
    config["name"] = data.get("name", config["name"])
    config["description"] = data.get("description", config["description"])
    config["absolute_path"] = data.get("absolute_path", config["absolute_path"])
    config["preprocessing"] = data.get("preprocessing", config["preprocessing"])
    
    return jsonify({"success": True, "config": config})


@app.route('/api/files/columns', methods=['POST'])
def get_file_columns():
    """Get columns from a file"""
    data = request.json
    file_path = data.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    columns, error = read_file_columns(file_path)
    
    if error:
        return jsonify({"error": error}), 400
    
    return jsonify({"columns": columns})


@app.route('/api/files/list', methods=['POST'])
def list_files():
    """List files in a directory"""
    data = request.json
    directory = data.get('directory')
    
    if not directory or not os.path.exists(directory):
        return jsonify({"error": "Directory not found"}), 404
    
    try:
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in ['.feather', '.parquet', '.csv']):
                files.append(filename)
        
        return jsonify({"files": sorted(files)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/config/file', methods=['POST'])
def add_file_config():
    """Add or update file configuration"""
    session_id = session.get('session_id')
    if not session_id or session_id not in configurations:
        return jsonify({"error": "No configuration found"}), 404
    
    data = request.json
    filename = data.get('filename')
    columns = data.get('columns', [])
    mapping = data.get('mapping')
    date_column = data.get('date_column')
    
    if not filename or not columns:
        return jsonify({"error": "Invalid file configuration"}), 400
    
    config = configurations[session_id]
    config["file_configs"][filename] = {
        "columns": columns,
        "mapping": mapping,
        "date_column": date_column
    }
    
    return jsonify({"success": True, "config": config})


@app.route('/api/config/file/<filename>', methods=['DELETE'])
def delete_file_config(filename):
    """Delete file configuration"""
    session_id = session.get('session_id')
    if not session_id or session_id not in configurations:
        return jsonify({"error": "No configuration found"}), 404
    
    config = configurations[session_id]
    if filename in config["file_configs"]:
        del config["file_configs"][filename]
    
    return jsonify({"success": True, "config": config})


@app.route('/api/export/json', methods=['GET'])
def export_json():
    """Export configuration as JSON"""
    session_id = session.get('session_id')
    if not session_id or session_id not in configurations:
        return jsonify({"error": "No configuration found"}), 404
    
    config = configurations[session_id]
    json_config = build_json_config(config)
    
    return jsonify(json_config)


@app.route('/api/export/python', methods=['GET'])
def export_python():
    """Export configuration as Python"""
    session_id = session.get('session_id')
    if not session_id or session_id not in configurations:
        return jsonify({"error": "No configuration found"}), 404
    
    config = configurations[session_id]
    python_config = build_python_config(config)
    
    return jsonify({"python": python_config})


@app.route('/api/download/json')
def download_json():
    """Download configuration as JSON file"""
    session_id = session.get('session_id')
    if not session_id or session_id not in configurations:
        return jsonify({"error": "No configuration found"}), 404
    
    config = configurations[session_id]
    json_config = build_json_config(config)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(json_config, f, indent=4)
        temp_path = f.name
    
    return send_file(temp_path, as_attachment=True, download_name='etl_config.json')


@app.route('/api/download/python')
def download_python():
    """Download configuration as Python file"""
    session_id = session.get('session_id')
    if not session_id or session_id not in configurations:
        return jsonify({"error": "No configuration found"}), 404
    
    config = configurations[session_id]
    python_config = build_python_config(config)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write(python_config)
        temp_path = f.name
    
    return send_file(temp_path, as_attachment=True, download_name='etl_config.py')


@app.route('/api/import', methods=['POST'])
def import_config():
    """Import configuration from JSON"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session found"}), 404
    
    data = request.json
    imported_config = data.get('config')
    
    if not imported_config:
        return jsonify({"error": "No configuration provided"}), 400
    
    # Parse the imported configuration
    config = {
        "name": imported_config.get("name", "amc_process"),
        "description": imported_config.get("description", "Preprocessing of AMC data"),
        "absolute_path": imported_config.get("absolute_path", ""),
        "preprocessing": imported_config.get("preprocessing", ""),
        "file_configs": {}
    }
    
    excluded_keys = ["name", "description", "absolute_path", "preprocessing", 
                    "final_cols", "tuple_vals_after", "tuple_vals_anybefore", 
                    "ref_date", "InitTransforms", "PreTransforms", "MergedTransforms"]
    
    for key, value in imported_config.items():
        if key not in excluded_keys and isinstance(value, list):
            file_config = {
                "columns": [],
                "mapping": None,
                "date_column": None
            }
            
            for item in value:
                if isinstance(item, str):
                    file_config["columns"].append(item)
                elif isinstance(item, dict):
                    for map_col, mappings in item.items():
                        if isinstance(mappings, dict):
                            file_config["mapping"] = {
                                "source_col": map_col,
                                "mapping_col": map_col,
                                "mappings": {map_col: mappings}
                            }
                            file_config["columns"].append(map_col)
                elif isinstance(item, list) and len(item) == 1:
                    file_config["date_column"] = item[0]
                    file_config["columns"].append(item[0])
            
            config["file_configs"][key] = file_config
    
    configurations[session_id] = config
    
    return jsonify({"success": True, "config": config})


@app.route('/api/clear', methods=['POST'])
def clear_config():
    """Clear all configurations"""
    session_id = session.get('session_id')
    if session_id in configurations:
        configurations[session_id] = {
            "name": "",
            "description": "",
            "absolute_path": "",
            "preprocessing": "",
            "files": {}
        }
    return jsonify({"success": True})


@app.route('/api/open-browser', methods=['POST'])
def open_file_browser():
    """Open the OS file browser at the specified path"""
    data = request.get_json()
    path = data.get('path', '')
    
    # If path is empty or doesn't exist, use home directory
    if not path or not os.path.exists(path):
        path = os.path.expanduser('~')
    
    # If path is a file, open its parent directory
    if os.path.isfile(path):
        path = os.path.dirname(path)
    
    try:
        system = platform.system()
        
        if system == 'Windows':
            os.startfile(path)
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', path], check=True)
        elif system == 'Linux':
            # Try different file managers in order of preference
            file_managers = [
                ['xdg-open', path],
                ['nautilus', path],
                ['dolphin', path],
                ['thunar', path],
                ['pcmanfm', path],
                ['nemo', path],
                ['caja', path]
            ]
            
            success = False
            for fm_cmd in file_managers:
                try:
                    subprocess.run(fm_cmd, check=True, stderr=subprocess.DEVNULL)
                    success = True
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if not success:
                return jsonify({
                    "success": False,
                    "error": "No file manager found. Please install xdg-utils or a file manager."
                }), 500
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported operating system: {system}"
            }), 500
        
        return jsonify({
            "success": True,
            "message": f"Opened file browser at: {path}"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/browse/directories', methods=['POST'])
def browse_directories():
    """List directories for path selection in web interface"""
    data = request.json
    path = data.get('path', os.path.expanduser('~'))
    
    if not os.path.exists(path):
        path = os.path.expanduser('~')
    
    # If it's a file, use its directory
    if os.path.isfile(path):
        path = os.path.dirname(path)
    
    try:
        items = []
        
        # Add parent directory option if not at root
        parent = os.path.dirname(path)
        if parent != path:
            items.append({
                'name': '📁 ..',
                'path': parent,
                'type': 'parent'
            })
        
        # List all items in the current path
        for item in sorted(os.listdir(path)):
            item_path = os.path.join(path, item)
            try:
                if os.path.isdir(item_path):
                    items.append({
                        'name': f'📁 {item}',
                        'path': item_path,
                        'type': 'directory'
                    })
                elif any(item.endswith(ext) for ext in ['.feather', '.parquet', '.csv']):
                    items.append({
                        'name': f'📄 {item}',
                        'path': item_path,
                        'type': 'file'
                    })
            except PermissionError:
                continue  # Skip items we can't access
        
        return jsonify({
            'success': True,
            'current_path': path,
            'items': items
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("=" * 60)
    print("ETL Configuration Builder Web Interface")
    print("=" * 60)
    print("\n🌐 Starting server at: http://localhost:2000")
    print("\n📝 Features:")
    print("   • Interactive file and column selection")
    print("   • Measurement mapping configuration")
    print("   • Date column wrapping")
    print("   • Export to JSON or Python format")
    print("   • Native OS file browser integration")
    print("\n⌨️  Press Ctrl+C to stop the server\n")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=2000)
