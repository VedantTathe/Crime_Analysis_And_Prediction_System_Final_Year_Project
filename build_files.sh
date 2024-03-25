#!/bin/bash

# echo "BUILD START"

# Navigate to the parent directory
# cd ..

# # Activate the virtual environment (update with your virtual environment activation command)
# source activate

# cd CrimeAnalysis
#!/bin/bash

echo "BUILD START"

# Activate the virtual environment (update with your virtual environment activation command)
source activate

# Navigate to the Django project directory

# Install project dependencies
pip install -r requirements.txt


# Collect static files
python manage.py collectstatic --noinput --clear


echo "BUILD END"
