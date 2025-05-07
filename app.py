from flask import Flask, request, jsonify, send_file
import os, requests, re, geopandas as gpd, zipfile, logging, traceback, rasterio
from shapely.geometry import Point
from werkzeug.utils import secure_filename
from flask_cors import CORS
import uuid
import psycopg2
from psycopg2.extras import DictCursor
import json
from datetime import datetime
import fiona
from shapely.geometry import shape
import tempfile
import shutil
from geo.Geoserver import Geoserver
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

DB_CONFIG = {
    'dbname': 'Geogit Intelli',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost'
}

# GeoServer configuration
GEOSERVER_CONFIG = {
    'url': 'http://10.7.236.23:8080/geoserver',
    'username': 'admin',
    'password': 'geoserver',
    'workspace': 'geogit'
}

# Directory to store uploaded raster files
RASTER_UPLOAD_DIR = 'raster_uploads'
os.makedirs(RASTER_UPLOAD_DIR, exist_ok=True)

# Voronoi configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-ae3d0fef8d0644e7a4090f41f0713a3c"  # Replace with your actual key

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

def ask_deepseek(query):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": query}]
    }

    try:
        logger.info("Sending request to DeepSeek API...")
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
        logger.debug(f"Received code from DeepSeek API: {code_blocks[0] if code_blocks else 'No code generated'}")
        return code_blocks[0] if code_blocks else None
    except Exception:
        logger.exception("Failed to get response from DeepSeek API.")
        return None

def find_file_by_ext(directory, ext):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(ext):
                return os.path.join(root, file)
    return None

@app.route('/')
def home():
    return 'Hello, Flask!'

@app.route('/api/projects', methods=['POST'])
def create_project():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'name' not in data:
            return jsonify({'error': 'Project name is required'}), 400
        
        # Generate a new UUID for the project
        project_id = str(uuid.uuid4())
        
        # Extract project data
        name = data['name']
        description = data.get('description', '')
        
        # Insert project into database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute(
            "INSERT INTO projects (id, name, description) VALUES (%s, %s, %s) RETURNING id, name, description, created_at",
            (project_id, name, description)
        )
        
        # Fetch the created project
        new_project = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        # Format the response
        response = {
            'id': str(new_project['id']),
            'name': new_project['name'],
            'description': new_project['description'],
            'created_at': new_project['created_at'].isoformat(),
            'message': 'Project created successfully'
        }
        
        return jsonify(response), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_id>/upload/vector', methods=['POST'])
def upload_vector_data(project_id):
    try:
        # Check if project exists
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
        project = cursor.fetchone()
        
        if not project:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Project not found'}), 404
        
        # Check if file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Generate version ID early so we can use it for the filename
        version_id = str(uuid.uuid4())
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save the file with version_id as name but keep original extension
        saved_filename = f"{version_id}{file_extension}"
        saved_file_path = os.path.join(uploads_dir, saved_filename)
        file.save(saved_file_path)
        
        # Process the file based on its type
        gdf = None
        file_format = None
        
        try:
            if file_extension == '.geojson':
                gdf = gpd.read_file(saved_file_path)
                file_format = 'geojson'
            elif file_extension == '.kml':
                gdf = gpd.read_file(saved_file_path, driver='KML')
                file_format = 'kml'
            elif file_extension == '.shp':
                gdf = gpd.read_file(saved_file_path)
                file_format = 'shapefile'
            elif file_extension == '.zip':
                # Extract the zip file to a subdirectory named after the version_id
                zip_extract_path = os.path.join(uploads_dir, version_id)
                os.makedirs(zip_extract_path, exist_ok=True)
                
                with zipfile.ZipFile(saved_file_path, 'r') as zip_ref:
                    zip_ref.extractall(zip_extract_path)
                
                # Look for shapefile in the extracted directory
                shp_files = []
                for root, dirs, files in os.walk(zip_extract_path):
                    for file in files:
                        if file.endswith('.shp'):
                            shp_files.append(os.path.join(root, file))
                
                if not shp_files:
                    raise ValueError("No shapefile found in the zip archive")
                
                # Use the first shapefile found
                gdf = gpd.read_file(shp_files[0])
                file_format = 'shapefile'
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Create commit message
            commit_message = f"Uploaded {file_format} file: {file.filename}"
            
            # Get the latest version number for this project
            cursor.execute(
                "SELECT version_number FROM versions WHERE project_id = %s ORDER BY created_at DESC LIMIT 1",
                (project_id,)
            )
            latest_version = cursor.fetchone()
            
            if latest_version:
                # Increment the version number
                try:
                    version_number = str(int(latest_version['version_number']) + 1)
                except ValueError:
                    # If version number is not a simple integer
                    version_number = latest_version['version_number'] + '.1'
            else:
                # First version
                version_number = '1'
            
            # Insert the new version
            cursor.execute(
                """
                INSERT INTO versions (id, project_id, version_number, commit_message, parent_version_id)
                VALUES (%s, %s, %s, %s, (
                    SELECT id FROM versions 
                    WHERE project_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT 1
                ))
                RETURNING id
                """,
                (version_id, project_id, version_number, commit_message, project_id)
            )
            
            # Update the project's current version
            cursor.execute(
                "UPDATE projects SET current_version_id = %s WHERE id = %s",
                (version_id, project_id)
            )
            
            # Store vector IDs to return in response
            vector_ids = []
            
            # Insert each feature into the vector_data table
            for _, row in gdf.iterrows():
                vector_id = str(uuid.uuid4())
                geom = row.geometry
                
                # Convert geometry to WKT format
                wkt = geom.wkt
                
                cursor.execute(
                    """
                    INSERT INTO vector_data (id, version_id, file_format, geometry, crs)
                    VALUES (%s, %s, %s, ST_GeomFromText(%s, 4326), %s)
                    """,
                    (vector_id, version_id, file_format, wkt, "EPSG:4326")
                )
                
                vector_ids.append(vector_id)
            
            # Add entry to versioning_history
            history_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO versioning_history (id, version_id, commit_message, data_type)
                VALUES (%s, %s, %s, %s)
                """,
                (history_id, version_id, commit_message, 'vector')
            )
            
            conn.commit()
            
            # Convert GeoDataFrame to GeoJSON for the response
            geojson_data = json.loads(gdf.to_json())
            
            # Return success response with geometry data
            return jsonify({
                'message': 'Vector data uploaded successfully',
                'project_id': project_id,
                'version_id': version_id,
                'version_number': version_number,
                'features_count': len(gdf),
                'geometry_data': geojson_data,
                'original_filename': file.filename,
                'saved_filename': saved_filename
            }), 201
            
        except Exception as e:
            # Roll back transaction in case of error
            conn.rollback()
            
            # Clean up uploaded file if something went wrong
            if os.path.exists(saved_file_path):
                os.remove(saved_file_path)
            
            # Clean up extracted zip directory if it exists
            zip_extract_path = os.path.join(uploads_dir, version_id)
            if os.path.exists(zip_extract_path):
                shutil.rmtree(zip_extract_path, ignore_errors=True)
            
            raise e
        
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/projects/<project_id>/upload/raster', methods=['POST'])
def upload_raster_data(project_id):
    try:
        # Check if project exists
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
        project = cursor.fetchone()
        
        if not project:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Project not found'}), 404
        
        # Check if file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Validate file extension
        if file_extension not in ['.tif', '.tiff']:
            return jsonify({'error': f'Unsupported file format: {file_extension}. Only .tif and .tiff files are supported.'}), 400
        
        try:
            # Generate a unique filename to avoid conflicts
            version_id = str(uuid.uuid4())  # Generate this first
            file_path = os.path.join(RASTER_UPLOAD_DIR, f"{version_id}.tif")
            
            # Save the file to the upload directory
            file.save(file_path)
            
            # Create a new version for this upload
          
            commit_message = f"Uploaded raster file: {file.filename}"
            
            # Get the latest version number for this project
            cursor.execute(
                "SELECT version_number FROM versions WHERE project_id = %s ORDER BY created_at DESC LIMIT 1",
                (project_id,)
            )
            latest_version = cursor.fetchone()
            
            if latest_version:
                # Increment the version number
                try:
                    version_number = str(int(latest_version['version_number']) + 1)
                except ValueError:
                    # If version number is not a simple integer
                    version_number = latest_version['version_number'] + '.1'
            else:
                # First version
                version_number = '1'
            
            # Insert the new version
            cursor.execute(
                """
                INSERT INTO versions (id, project_id, version_number, commit_message, parent_version_id)
                VALUES (%s, %s, %s, %s, (
                    SELECT id FROM versions 
                    WHERE project_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT 1
                ))
                RETURNING id
                """,
                (version_id, project_id, version_number, commit_message, project_id)
            )
            
            # Update the project's current version
            cursor.execute(
                "UPDATE projects SET current_version_id = %s WHERE id = %s",
                (version_id, project_id)
            )
            
            # Initialize GeoServer connection
            geo = Geoserver(
                GEOSERVER_CONFIG['url'],
                username=GEOSERVER_CONFIG['username'],
                password=GEOSERVER_CONFIG['password']
            )
            
            # Ensure workspace exists
            try:
                geo.create_workspace(workspace=GEOSERVER_CONFIG['workspace'])
            except Exception as e:
                # Workspace might already exist, continue
                pass
            
            # Generate a unique layer name
            layer_name = f"raster_{version_id}"
            
            # Upload raster to GeoServer
            geo.create_coveragestore(
                layer_name=f"raster_{version_id}",  # Simplified layer name
                path=file_path,
                workspace=GEOSERVER_CONFIG['workspace']
            )
            
            # Construct the WMS URL and fetch bounding box
            wms_url = create_wms_url(GEOSERVER_CONFIG['workspace'], layer_name)
            bbox = fetch_bounding_box(wms_url, layer_name)
            
            # Construct the Mapbox URL
            mapbox_url = create_mapbox_url(GEOSERVER_CONFIG['workspace'], layer_name)
            
            # Insert raster data into database
            raster_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO raster_data (id, version_id, wms_url, file_format)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (raster_id, version_id, wms_url, file_extension[1:])  # Remove the dot from extension
            )
            
            # Add entry to versioning_history
            history_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO versioning_history (id, version_id, commit_message, data_type)
                VALUES (%s, %s, %s, %s)
                """,
                (history_id, version_id, commit_message, 'raster')
            )
            
            conn.commit()
            
            # Return success response with Mapbox URL and bounding box
            return jsonify({
                'message': 'Raster data uploaded successfully',
                'project_id': project_id,
                'version_id': version_id,
                'version_number': version_number,
                'mapbox_url': mapbox_url,
                'bounding_box': bbox,
                'file_path': file_path
            }), 201
            
        except Exception as e:
            # Roll back transaction in case of error
            conn.rollback()
            # Clean up the file if it was saved
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            raise e
        
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_wms_url(workspace, layer_name):
    """
    Create the WMS URL for the published layer in GeoServer to get bounding box.
    """
    wms_url = f"http://10.7.236.23:8080/geoserver/{workspace}/wms?service=WMS&request=GetCapabilities"
    return wms_url

def create_mapbox_url(workspace, store_name):
    """
    Create the Mapbox URL for the published layer in GeoServer.
    """
    mapbox_url = (
        f"http://10.7.236.23:8080/geoserver/wms?service=WMS&request=GetMap"
        f"&layers={workspace}:{store_name}&styles=&format=image/png"
        f"&transparent=true&version=1.1.1&width=256&height=256"
        f"&srs=EPSG:3857&bbox={{bbox-epsg-3857}}"
    )
    return mapbox_url

def fetch_bounding_box(wms_url, layer_name):
    """
    Fetch the bounding box of the layer from the WMS GetCapabilities response.
    """
    response = requests.get(wms_url)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        namespace = {'wms': 'http://www.opengis.net/wms'}
        for layer in root.findall(".//wms:Layer", namespace):
            name = layer.find("wms:Name", namespace)
            if name is not None and name.text == layer_name:
                bbox = layer.find("wms:BoundingBox[@CRS='CRS:84']", namespace)
                if bbox is not None:
                    minx = bbox.get("minx")
                    miny = bbox.get("miny")
                    maxx = bbox.get("maxx")
                    maxy = bbox.get("maxy")
                    bounding_box = {
                        'minx': minx,
                        'miny': miny,
                        'maxx': maxx,
                        'maxy': maxy
                    }
                    return bounding_box
        return {'error': f'Layer {layer_name} not found in capabilities.'}
    else:
        return {'error': f"Failed to retrieve WMS capabilities. Status code: {response.status_code}"}

@app.route('/voronoi', methods=['POST'])
def voronoi():
    try:
        # Check required parameters
        if 'file_inputs' not in request.json or 'prompt' not in request.json or 'project_id' not in request.json:
            logger.warning("Missing required parameters in request.")
            return jsonify({"error": "Missing file_inputs, prompt, or project_id"}), 400

        project_id = request.json['project_id'] 
        file_inputs = request.json['file_inputs']
        prompt = request.json['prompt']        
        prompt += ("Use GeoPandas, rasterio, Shapely if applicable. "
            "Don't plot. Save to file. Give Only python code. No extra text. Stay within max tries. "
            "{If the task above is about clipping then 'Clip to the actual shape of the raster's valid data (i.e., mask out areas where the raster has no data, not just the bounding box)', use this approach with rasterio.open('') as src: raster_crs = src.crs mask_data = src.dataset_mask() shapes = shapes(mask_data, transform=src.transform)  geom = [Polygon(shape[0]['coordinates'][0]) for shape in shapes if shape[1] == 255][0],  gdf_voronoi = gpd.read_file('') gdf_voronoi_clipped = gdf_voronoi[gdf_voronoi.intersects(geom)].copy() gdf_voronoi_clipped['geometry'] = gdf_voronoi_clipped.intersection(geom) gdf_voronoi_clipped.crs = raster_crs}. "
            "{ If the task above is about generating voronoi map, 'use voronoi_polygons = voronoi_diagram(points)', Beware of this error: TypeError: 'Polygon' object is not iterable "
            "{If the task above is about calculating the total population within each polygon using the raster values,for those polygons who have null in their population column after the population is computed, add random values between 100 to 1000 in population column for those}"
            "{If the task is related to buffer, make sure to avoid this error:  Geometry is in a geographic CRS. Results from 'buffer' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.}"
            "{If the task is to get data from OSM, use 'features_from_place' instead of 'geometries_from_place', and make sure data downloaded is saved in geojson}"
            "{If the task is related to downloading building footprints from OSM, # Filter to only polygons and ensure valid geometries, and then export to geojson}"
            "{If the task is related to downloading amenities, # Filter to only Point and ensure valid geometries, and then export to geojson, similarly filter to point, line or polygon, based on what user has asked in the task}"
            "If the task is related to downloading something from OSM, # Filter to point, line or polygon, and ensure valid geometries, based on what user has asked in the task.")

        logger.debug(f"Processing request with prompt: {prompt}")

        # Determine file paths based on data type
        uploaded_files = {}
        output_path = None

        for idx, entry in enumerate(file_inputs):
            fid = entry.get("id")
            ftype = entry.get("type")

            if not fid or not ftype:
                logger.error(f"Invalid file input: {entry}")
                return jsonify({"error": f"Each file input must have 'id' and 'type': {entry}"}), 400

            if ftype == 'vector':
                vector_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{fid}.geojson")
                if not os.path.exists(vector_path):
                    logger.error(f"Vector file not found: {vector_path}")
                    return jsonify({"error": f"Vector file not found: {fid}"}), 404
                uploaded_files[f"vector_{idx}"] = vector_path
                logger.info(f"Using vector file: {vector_path}")

            elif ftype == 'raster':
                raster_path = os.path.join(RASTER_UPLOAD_DIR, f"{fid}.tif")
                if not os.path.exists(raster_path):
                    logger.error(f"Raster file not found: {raster_path}")
                    return jsonify({"error": f"Raster file not found: {fid}"}), 404
                uploaded_files[f"raster_{idx}"] = raster_path
                logger.info(f"Using raster file: {raster_path}")

            else:
                logger.error(f"Invalid file type: {ftype}")
                return jsonify({"error": f"Invalid file type: {ftype}"}), 400
        
        code = ask_deepseek(prompt)
        logger.info(f"Generated code: {code}")
        if not code:
            logger.error("Code generation failed.")
            return jsonify({"error": "Code generation failed."}), 500
        
        try:
            tif_matches = re.findall(r'["\']([^"\']+\.tif)["\']', code, re.IGNORECASE)
            shp_matches = re.findall(r'["\']([^"\']+\.shp)["\']', code, re.IGNORECASE)
            geojson_matches = re.findall(r'["\']([^"\']+\.geojson)["\']', code, re.IGNORECASE)
            output_matches = re.findall(r'["\']([^"\']+\.(?:shp|tif|geojson|csv))["\']', code, re.IGNORECASE)

            for match in tif_matches:
                for _, actual_path in uploaded_files.items():
                    if actual_path.lower().endswith('.tif'):
                        safe_path = actual_path.replace("\\", "/")
                        code = code.replace(match, safe_path)

            for match in shp_matches:
                for _, actual_path in uploaded_files.items():
                    if actual_path.lower().endswith('.shp'):
                        for line in code.splitlines():
                            if match in line and 'read_file' in line:
                                safe_path = actual_path.replace("\\", "/")
                                code = code.replace(match, safe_path)

            for match in geojson_matches:
                for _, actual_path in uploaded_files.items():
                    if actual_path.lower().endswith('.geojson'):
                        for line in code.splitlines():
                            if match in line and 'read_file' in line:
                                safe_path = actual_path.replace("\\", "/")
                                code = code.replace(match, safe_path)

            output_basename = None
            output_base_ext = None
            output_path = None
            for match in output_matches:
                for line in code.splitlines():
                    if match in line and 'to_file' in line:
                        if any(ext in match.lower() for ext in [".shp", ".geojson"]):
                            output_basename = os.path.splitext(os.path.basename(match))[0]
                            output_base_ext = os.path.splitext(os.path.basename(match))[1].lower()
                            output_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(match))
                            safe_path = output_path.replace("\\", "/")
                            code = code.replace(match, safe_path)

            logger.debug(code)
            exec_globals = {"gpd": gpd, "rasterio": rasterio}
            exec(code, exec_globals)
            logger.info("Geospatial code executed successfully.")
            
          

            # Check if output file was created
            if not output_path or not os.path.exists(output_path):
                logger.error("Output file not generated")
                return jsonify({"error": "Output file not generated"}), 500
                
            # Determine if this is a vector or raster output
            if output_path.lower().endswith(('.tif', '.tiff')):
                # Upload the generated file to the project as raster
                with open(output_path, 'rb') as f:
                    files = {'file': (os.path.basename(output_path), f)}
                    upload_response = requests.post(
                        f"{request.host_url}api/projects/{project_id}/upload/raster",
                        files=files
                    )
                
                if upload_response.status_code != 201:
                    logger.error(f"Failed to upload raster data: {upload_response.text}")
                    return jsonify({"error": "Failed to upload raster data"}), 500
                
                upload_data = upload_response.json()
                
                return jsonify({
                    "id": upload_data.get('version_id'),
                    "name": os.path.basename(output_path),
                    "type": "raster",
                    "format": output_base_ext[1:],  # "tif" or "tiff"
                    "crs": "EPSG:4326",
                    "status": "new",
                    "version_number": upload_data.get('version_number'),
                    "mapbox_url": upload_data.get('mapbox_url'),
                    "bounding_box": upload_data.get('bounding_box'),
                    "file_path": upload_data.get('file_path')
                }), 200
            else:
                # This is the existing vector data handling (geojson/shp)
                with open(output_path, 'rb') as f:
                    files = {'file': (os.path.basename(output_path), f)}
                    upload_response = requests.post(
                        f"{request.host_url}api/projects/{project_id}/upload/vector",
                        files=files
                    )
                
                if upload_response.status_code != 201:
                    logger.error(f"Failed to upload vector data: {upload_response.text}")
                    return jsonify({"error": "Failed to upload vector data"}), 500
                
                upload_data = upload_response.json()
                
                # Read the output file to get geometry data
                gdf = gpd.read_file(output_path)
                geojson_data = json.loads(gdf.to_json())
                
                return jsonify({
                    "id": upload_data.get('version_id'),
                    "name": os.path.basename(output_path),
                    "type": "vector",
                    "format": output_base_ext[1:],  # "geojson" or "shp"
                    "crs": "EPSG:4326",
                    "status": "new",
                    "version_number": upload_data.get('version_number'),
                    "geometry_data": geojson_data,
                    "features_count": len(gdf)
                }), 200

        except Exception as e:
            logger.error(f"Error during geospatial code execution: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.exception("Processing failed.")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)