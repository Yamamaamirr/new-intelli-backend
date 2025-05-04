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

# Database connection configuration
DB_CONFIG = {
    'dbname': 'Geogit Intelli',
    'user': 'postgres',
    'password': 'masood73',
    'host': 'localhost'
}

# GeoServer configuration
GEOSERVER_CONFIG = {
    'url': 'http://127.0.0.1:8080/geoserver',
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
        
        # Create a temporary directory to store the uploaded file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)
        
        # Process the file based on its type
        gdf = None
        file_format = None
        
        try:
            if file_extension == '.geojson':
                gdf = gpd.read_file(temp_file_path)
                file_format = 'geojson'
            elif file_extension == '.kml':
                gdf = gpd.read_file(temp_file_path, driver='KML')
                file_format = 'kml'
            elif file_extension == '.shp':
                gdf = gpd.read_file(temp_file_path)
                file_format = 'shapefile'
            elif file_extension == '.zip':
                # Extract the zip file
                zip_extract_path = os.path.join(temp_dir, 'extracted')
                os.makedirs(zip_extract_path, exist_ok=True)
                
                with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
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
            
            # Create a new version for this upload
            version_id = str(uuid.uuid4())
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
                'geometry_data': geojson_data
            }), 201
            
        except Exception as e:
            # Roll back transaction in case of error
            conn.rollback()
            raise e
        
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
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
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(RASTER_UPLOAD_DIR, unique_filename)
            
            # Save the file to the upload directory
            file.save(file_path)
            
            # Create a new version for this upload
            version_id = str(uuid.uuid4())
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
            layer_name = f"raster_{project_id}_{version_number}".replace('-', '_').replace('.', '_')
            
            # Upload raster to GeoServer
            geo.create_coveragestore(
                layer_name=layer_name,
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
    wms_url = f"http://10.7.237.121:8080/geoserver/{workspace}/wms?service=WMS&request=GetCapabilities"
    return wms_url

def create_mapbox_url(workspace, store_name):
    """
    Create the Mapbox URL for the published layer in GeoServer.
    """
    mapbox_url = (
        f"http://10.7.237.121:8080/geoserver/wms?service=WMS&request=GetMap"
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
        if 'prompt' not in request.form:
            logger.warning("Missing prompt in request.")
            return jsonify({"error": "Missing files or prompt"}), 400

        files = request.files.getlist("files")
        prompt = request.form['prompt']
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

        print(prompt)
        logger.debug(prompt)
        tif_path, shp_dir, geojson_path = None, None, None
        

        for f in files:
            filename = secure_filename(f.filename)
            file_ext = os.path.splitext(filename)[1].lower()
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(save_path)
            print("save_path",save_path)

            logger.info(f"File uploaded: {save_path}")

            if file_ext == '.zip':
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    extracted_names = zip_ref.namelist()
                    zip_ref.extractall(app.config['UPLOAD_FOLDER'])

                # Find the top-level folder inside the zip, if any
                top_dirs = {name.split('/')[0] for name in extracted_names if '/' in name}
                if top_dirs:
                    extracted_folder = os.path.join(app.config['UPLOAD_FOLDER'], list(top_dirs)[0])
                else:
                    extracted_folder = app.config['UPLOAD_FOLDER']
                
                shp_dir = extracted_folder
                logger.info(f"Extracted ZIP folder path: {shp_dir}")
            elif file_ext == '.tif':
                tif_path = save_path
            elif file_ext == '.geojson':
                geojson_path = save_path

        uploaded_files = {}

        if shp_dir:
            shapefile = find_file_by_ext(shp_dir, '.shp')
            if shapefile:
                uploaded_files["shapefile"] = shapefile

        if tif_path:
            uploaded_files["raster"] = tif_path

        if geojson_path:
            uploaded_files["geojson"] = geojson_path


        if not uploaded_files:
            logger.error("No valid .tif or .shp or .geojson found in upload.")
            #return jsonify({"error": "No valid .tif or .shp found in upload."}), 400

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

            logger.debug(f"Found .tif references in code: {tif_matches}")
            logger.debug(f"Found .shp references in code: {shp_matches}")
            logger.debug(f"Found .geojson references in code: {geojson_matches}")
            logger.debug(f"Found output references in code: {output_matches}")

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
        except Exception as e:
            logger.error(f"Error during geospatial code execution: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
        
        output_folder = app.config['OUTPUT_FOLDER']
        if output_base_ext == '.geojson':
            output_geojson_file = os.path.join(output_folder, output_basename + '.geojson')
            if os.path.exists(output_geojson_file):
                return send_file(output_geojson_file, as_attachment=True)
            else:
                return jsonify({"error": "GeoJSON output file not found."}), 404

        elif output_base_ext == '.shp':
            zip_output_path = os.path.join(output_folder, output_basename + '.zip')
            with zipfile.ZipFile(zip_output_path, 'w') as zipf:
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    filepath = os.path.join(output_folder, output_basename + ext)
                    if os.path.exists(filepath):
                        zipf.write(filepath, os.path.basename(filepath))
            return send_file(zip_output_path, as_attachment=True)

        else:
            return jsonify({"error": "Unsupported output file type."}), 400

    except Exception as e:
        logger.exception("Processing failed.")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)