from typing import Dict
import psycopg2
from config.db import (
    postgresql_host,
    postgresql_db,
    postgresql_password,
    postgresql_user
)
from schemas.schema import CameraConfig

def get_cameras_from_db() -> Dict[str, CameraConfig]:
    """
    Fetch camera configurations from PostgreSQL database
    """
    conn = psycopg2.connect(
        dbname=postgresql_db,
        user=postgresql_user,
        password=postgresql_password,
        host=postgresql_host
    )
    cur = conn.cursor()
    
    # Query to get active streams
    cur.execute("""
        SELECT 
            stream_name,
            stream_url,
            stream_gate_id,
            stream_direction,
            stream_debug
        FROM "tblStream"
        WHERE stream_status = 'A'
        ORDER BY stream_gate_id, stream_direction
    """)
    
    streams = cur.fetchall()
    cameras = {}
    
    for idx, stream in enumerate(streams, 1):
        stream_name, stream_url, gate_id, direction, debug = stream
        
        # Extract username and password from RTSP URL if present
        username = None
        password = None
        if '@' in stream_url:
            try:
                credentials = stream_url.split('@')[0].split('//')[1]
                username, password = credentials.split(':')
            except IndexError:
                pass  # Invalid URL format, skip credentials
        
        camera_config = CameraConfig(
            name=stream_name,
            url=stream_url,
            gate_id=str(gate_id),
            direction=direction.lower(),
            username=username,
            password=password,
            enabled=not debug  # Use debug flag as opposite of enabled
        )
        
        cameras[f"camera{idx}"] = camera_config
    
    cur.close()
    conn.close()
    
    return cameras