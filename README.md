# Thunderbird Email Client Plugin and Classifier 

## Features

- **Automatic Email Classification**: ML Models to classify emails.
- **User-Friendly Interface**: Easy to use with a minimal learning curve.

## Installation

### To install on your Thunderbird client:

1. Open Thunderbird and go to `Add-ons and Themes`.
2. Click on the gear icon and select `Install Add-on From File`.
3. Navigate to the downloaded `.xpi` file and select it.
4. Follow the prompts to install the add-on.

### To install Backend:

1. To access the BERT Model contact the admin (vamsy679)
2. Place file in the Models folder in the backend folder.
3. Create virtual env and activate the venv.
4. Install the requirements `pip install requirements.txt`.
5. Run the FASTAPI using `uvicorn main:app --host 127.0.0.1 --port 8000`.

## Usage
### Checking Email Classification

To test the Model's effectiveness:

- Select several emails that have not been used for training.
- Right-click and choose `Classify Email` from the EClassify menu to see the automatic classification in action.
