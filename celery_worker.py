from celery import Celery
import os
import importlib.util

# Create output directory for celery files
CELERY_DIR = "celery_files"
os.makedirs(CELERY_DIR, exist_ok=True)

# Configure Celery to use filesystem as broker instead of Redis
# This is simpler to set up when Redis isn't available
celery_app = Celery(
    'transcription_app',
    broker=f'filesystem://{os.path.abspath(CELERY_DIR)}',
    backend=f'file://{os.path.abspath(CELERY_DIR)}/results'
)

# Additional broker settings for filesystem
celery_app.conf.update(
    broker_transport_options={
        'data_folder_in': os.path.join(CELERY_DIR, 'in'),
        'data_folder_out': os.path.join(CELERY_DIR, 'out'),
        'data_folder_processed': os.path.join(CELERY_DIR, 'processed')
    },
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=10800,  # 3 hours max task runtime
    worker_prefetch_multiplier=1,  # Process one task at a time
    result_expires=86400,  # Results expire after 1 day
)

# Create necessary directories for filesystem broker
for folder in ['in', 'out', 'processed', 'results']:
    os.makedirs(os.path.join(CELERY_DIR, folder), exist_ok=True)

# Explicitly load app.py instead of using autodiscover
print("Manually importing app.py...")
spec = importlib.util.spec_from_file_location("app", "./app.py")
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
print("Successfully imported app.py")

# No need to use autodiscover_tasks or import app directly anymore

if __name__ == '__main__':
    celery_app.start() 