import time
from datetime import datetime

import psycopg2

DB_CONFIG = {
    "dbname": "zulip_events",
    "user": "zulip_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5434"
}

def store_event(recipient, content, operation_type, updating_event_id=None):
    """Сохраняет событие (отправка или редактирование) в базу данных."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO events (message_content, recipient, status, operation_type, created_at, updating_event_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (content, recipient, 'pending', operation_type, datetime.now(), updating_event_id))

    event_id = cursor.fetchone()[0]  # Получаем ID события, который будем использовать для редактирования
    conn.commit()
    cursor.close()
    conn.close()

    return event_id  # Возвращаем ID события


def get_message_id_from_event(event_id):
    """Получаем message_id из базы данных по event_id."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT message_id
        FROM events
        WHERE id = %s;
    """, (event_id,))

    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        return result[0]  # Возвращаем message_id
    return None  # Если не найдено, возвращаем None


def get_pending_events():
    """Получаем события со статусом 'pending' из базы данных."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
            SELECT id, message_content, recipient, operation_type, updating_event_id
            FROM events
            WHERE status = 'pending'
            order by created_at;
        """)

    events = cursor.fetchall()
    cursor.close()
    conn.close()

    return [{'id': e[0], 'message_content': e[1], 'recipient': e[2], 'operation_type': e[3], 'updating_event_id': e[4]} for e in events]


def update_event_with_message_id(event_id, message_id):
    """Обновляем событие с message_id в базе данных."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
            UPDATE events
            SET message_id = %s
            WHERE id = %s;
        """, (message_id, event_id))

    conn.commit()
    cursor.close()
    conn.close()

def update_event_status(event_id, status):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
                UPDATE events
                SET status = %s
                WHERE id = %s;
            """, (status, event_id))

    conn.commit()
    cursor.close()
    conn.close()
