import sqlite3


DATABASE_PATH = "plate_database/traffic_tracking.db"


print("=" * 70)
print("SMART TRAFFIC MONITORING SYSTEM")
print("PERSON LAST-SEEN SEARCH")
print("=" * 70)


# ============================================================
# CONNECT TO DATABASE
# ============================================================

try:
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

except Exception as e:
    print("\nERROR connecting to database:")
    print(e)
    exit()


# ============================================================
# CHECK AVAILABLE PERSONS
# ============================================================

cursor.execute("""
    SELECT DISTINCT person_id
    FROM person_tracking
    ORDER BY person_id
""")

persons = cursor.fetchall()


print("\nAvailable tracked persons:")

if not persons:
    print("No persons found in database.")
    conn.close()
    exit()

for person in persons:
    print(f"  Person_{person[0]}")


# ============================================================
# USER INPUT
# ============================================================

user_input = input(
    "\nEnter Person ID "
    "(example: Person_1): "
).strip()


# Remove "Person_" if user enters it

if user_input.lower().startswith("person_"):
    user_input = user_input[7:]


try:
    person_id = int(user_input)

except ValueError:

    print("\nInvalid Person ID.")
    conn.close()
    exit()


# ============================================================
# SEARCH DATABASE
# ============================================================

print("\nSearching database...")


cursor.execute("""
    SELECT
        person_id,
        camera_id,
        city,
        road,
        timestamp,
        frame,
        confidence,
        video_source
    FROM person_tracking
    WHERE person_id = ?
    ORDER BY frame ASC
""", (person_id,))


records = cursor.fetchall()


# ============================================================
# PERSON NOT FOUND
# ============================================================

if not records:

    print("\n" + "=" * 70)
    print("PERSON SEARCH RESULT")
    print("=" * 70)

    print(f"\nPerson_{person_id}")
    print("Status : NOT FOUND")

    print("\nThis person was not found in the tracking database.")

    print("=" * 70)

    conn.close()
    exit()


# ============================================================
# PERSON FOUND
# ============================================================

print("\n" + "=" * 70)
print("PERSON SEARCH RESULT")
print("=" * 70)

print(f"\nPerson ID    : Person_{person_id}")
print("Status       : FOUND")
print(f"Observations : {len(records)}")


# ============================================================
# ALL OBSERVATIONS
# ============================================================

print("\n" + "-" * 70)
print("TRACKING HISTORY")
print("-" * 70)


for index, record in enumerate(records, start=1):

    (
        pid,
        camera,
        city,
        road,
        timestamp,
        frame,
        confidence,
        video
    ) = record

    print(f"\nObservation #{index}")

    print(f"Camera      : {camera}")
    print(f"City        : {city}")
    print(f"Road        : {road}")
    print(f"Timestamp   : {timestamp}")
    print(f"Frame       : {frame}")
    print(f"Confidence  : {confidence:.2f}")
    print(f"Video       : {video}")


# ============================================================
# FIRST SEEN
# ============================================================

first_seen = records[0]

print("\n" + "=" * 70)
print("FIRST SEEN")
print("=" * 70)

print(f"Person      : Person_{person_id}")
print(f"Camera      : {first_seen[1]}")
print(f"City        : {first_seen[2]}")
print(f"Road        : {first_seen[3]}")
print(f"Timestamp   : {first_seen[4]}")
print(f"Frame       : {first_seen[5]}")
print(f"Video       : {first_seen[7]}")


# ============================================================
# LAST SEEN
# ============================================================

last_seen = records[-1]

print("\n" + "=" * 70)
print("LAST SEEN LOCATION")
print("=" * 70)

print(f"Person      : Person_{person_id}")
print(f"Camera      : {last_seen[1]}")
print(f"City        : {last_seen[2]}")
print(f"Road        : {last_seen[3]}")
print(f"Timestamp   : {last_seen[4]}")
print(f"Frame       : {last_seen[5]}")
print(f"Confidence  : {last_seen[6]:.2f}")
print(f"Video       : {last_seen[7]}")


# ============================================================
# CAMERA HISTORY
# ============================================================

print("\n" + "=" * 70)
print("CAMERA HISTORY")
print("=" * 70)


cursor.execute("""
    SELECT camera_id, COUNT(*)
    FROM person_tracking
    WHERE person_id = ?
    GROUP BY camera_id
    ORDER BY camera_id
""", (person_id,))


camera_history = cursor.fetchall()


for camera, count in camera_history:

    print(
        f"{camera:<15} : "
        f"{count} observation(s)"
    )


# ============================================================
# ROAD HISTORY
# ============================================================

print("\n" + "=" * 70)
print("ROAD HISTORY")
print("=" * 70)


cursor.execute("""
    SELECT road, COUNT(*)
    FROM person_tracking
    WHERE person_id = ?
    GROUP BY road
    ORDER BY road
""", (person_id,))


road_history = cursor.fetchall()


for road, count in road_history:

    print(
        f"{road:<20} : "
        f"{count} observation(s)"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("PERSON SEARCH COMPLETED")
print("=" * 70)


conn.close()