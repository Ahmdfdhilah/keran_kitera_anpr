import psycopg2
from config.db import (
    postgresql_host,
    postgresql_db,
    postgresql_password,
    postgresql_user
)


# Fungsi untuk mengambil vehicle_id berdasarkan vehicle_number_plate
def find_vehicle_id_by_number_plate(number_plate):
    conn = psycopg2.connect(
        dbname=postgresql_db,
        user=postgresql_user,
        password=postgresql_password,
        host=postgresql_host
    )
    cur = conn.cursor()
    cur.execute(
        """SELECT vehicle_id FROM public."tblVehicle" WHERE vehicle_number_plate = %s""", (number_plate,)
    )
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else None

# Fungsi untuk mengambil vehicle_id berdasarkan vehicle_rfid_sticker
def find_vehicle_id_by_rfid_sticker(rfid_sticker):
    conn = psycopg2.connect(
        dbname=postgresql_db,
        user=postgresql_user,
        password=postgresql_password,
        host=postgresql_host
    )
    cur = conn.cursor()
    cur.execute(
        """SELECT vehicle_id FROM public."tblVehicle" WHERE vehicle_rfid_sticker = %s""", (rfid_sticker,)
    )
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else None

# Fungsi untuk mengambil vehicle_id berdasarkan user_rfid_card
def find_vehicle_id_by_user_rfid_card(user_rfid_card):
    conn = psycopg2.connect(
        dbname=postgresql_db,
        user=postgresql_user,
        password=postgresql_password,
        host=postgresql_host
    )
    cur = conn.cursor()
    query = f"""
        SELECT v.vehicle_id 
        FROM public."tblVehicle" v 
        JOIN public."tblUser" u ON v.vehicle_user_id = u.user_id 
        WHERE u.user_rfid_card = %s
    """
    cur.execute(query, (user_rfid_card,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else None
