# EClassify

## Features

- **Automatic Email Classification**: Leverages ML Models to classify emails.
- **User-Friendly Interface**: Easy to use with a minimal learning curve.

## Installation

### To install EClassify on your Thunderbird client:

1. Open Thunderbird and go to `Add-ons and Themes`.
2. Click on the gear icon and select `Install Add-on From File`.
3. Navigate to the downloaded `.xpi` file and select it.
4. Follow the prompts to install the add-on.

### To install EClassify Backend:

1. Download the BERT Model from here https://drive.google.com/file/d/1zIOcVWpkB5aFEb3S3yEAAIc_LasyByKT/view?usp=drive_link.
2. Place it in the Models folder in the backend folder.
3. Create virtual env and activate the venv.
4. Install the requirements `pip install requirements.txt`.
5. Run the FASTAPI using `uvicorn main:app --host 127.0.0.1 --port 8000`.

## Usage
### Checking Email Classification

To test the Model's effectiveness:

- Select several emails that have not been used for training.
- Right-click and choose `Classify Email` from the EClassify menu to see the automatic classification in action.
