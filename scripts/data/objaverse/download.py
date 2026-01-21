import objaverse
import random

# Download Objaverse objects by random UIDs
print("Loading UIDs...")
uids = objaverse.load_uids()
print(f"Total objects available: {len(uids)}")
random_uids = random.sample(uids, 10)
print("Downloading objects...")
objects = objaverse.load_objects(
    uids=random_uids,
    download_processes=4  # Number of parallel downloads
)
for uid, path in objects.items():
    print(f"UID: {uid} -> Path: {path}")
