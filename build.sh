#!/bin/bash
set -o errexit

pip install -r requirements.txt

cd training_feedback_system

python manage.py collectstatic --no-input
python manage.py migrate

echo "Build completed successfully!"
