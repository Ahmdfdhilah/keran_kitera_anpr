from redis import Redis
from config.db import (
    redis_db,
    redis_host,
    redis_port
)


r = Redis(host=redis_host, port=redis_port, db=redis_db)

def save_card_reading(vehicle_id, card_data):
    key = f"card_readings:{vehicle_id}"
    r.sadd(key, card_data)

def save_sticker_reading(vehicle_id, sticker_data):
    key = f"sticker_readings:{vehicle_id}"
    r.sadd(key, sticker_data)

def save_anpr_reading(vehicle_id, anpr_data):
    key = f"anpr_readings:{vehicle_id}"
    r.sadd(key, anpr_data)

def check_vehicle_readings(vehicle_id):
    card_readings = r.smembers(f"card_readings:{vehicle_id}")
    sticker_readings = r.smembers(f"sticker_readings:{vehicle_id}")
    anpr_readings = r.smembers(f"anpr_readings:{vehicle_id}")

    matching_readings = card_readings.intersection(sticker_readings, anpr_readings)

    if matching_readings:
        # Mobil memiliki semua jenis pembacaan, kirim sinyal untuk membuka gerbang
        print(f"Vehicle with ID {vehicle_id} is allowed to pass. Opening gate...")

        # Hapus data pembacaan setelah berhasil
        r.delete(f"card_readings:{vehicle_id}")
        r.delete(f"sticker_readings:{vehicle_id}")
        r.delete(f"anpr_readings:{vehicle_id}")
        return True
    
    print(f"Vehicle with ID {vehicle_id} does not have complete readings. Gate remains closed.")
    return False
