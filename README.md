# Smishing Detection Project

This project focuses on detecting smishing (SMS phishing) using machine learning and adversarial training techniques.

# 1. Python Project
## Project Structure
- `code/` : Source code for data preparation, model architecture, adversarial training, and main scripts.
- `data/` : Raw and processed datasets, adversarial examples, and logs.
- `model/` : Exported models and related files.
- `plots/` : Training and evaluation plots.
- `requirements.txt` : Python dependencies for the project.

## Development Environment Setup

### 1. Install Python 3.9.23
Ensure you have Python 3.9.23 installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/release/python-3923/).

### 2. Create a Virtual Environment
It is recommended to use a virtual environment named `smishing_env` to manage dependencies.

Open PowerShell and run:
```powershell
python -m venv smishing_env
```

### 3. Activate the Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  .\smishing_env\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .\smishing_env\Scripts\activate.bat
  ```
- **Linux/MacOS:**
  ```bash
  source smishing_env/bin/activate
  ```

### 4. Upgrade pip (Recommended)
```powershell
python -m pip install --upgrade pip
```

### 5. Install Project Dependencies
Install all required packages using the provided `requirements.txt` file:
```powershell
pip install -r python/requirements.txt
```

# 2. Android Project
## Project Structure
- `android/` : Main Android project directory
  - `app/` : Application module (source code, resources, manifest)
  - `build.gradle` : Project-level Gradle build file
  - `gradle.properties` : Gradle configuration properties
  - `gradlew`, `gradlew.bat` : Gradle wrapper scripts
  - `settings.gradle` : Gradle settings
  - `gradle/` : Gradle wrapper and version catalog
    - `libs.versions.toml` : Dependency versions
    - `wrapper/` : Gradle wrapper files

## Android Environment Setup

### 1. Install Android Studio
Download and install [Android Studio](https://developer.android.com/studio) (latest stable version recommended).

### 2. Open the Project
- Launch Android Studio.
- Select "Open an Existing Project" and choose the `android/` directory.

### 3. Build the Project
- Let Gradle sync and download dependencies automatically.
- If prompted, install any missing SDK components.

### 4. Run the App
- Connect an Android device or start an emulator.
- Click the Run button (▶️) in Android Studio to build and launch the app.

## Additional Notes
- All Python code is located in the `code/` directory.
- Data files are in the `data/` directory.
- Model files and plots are in the `model/` and `plots/` directories respectively.
- Android app source and configuration are in the `android/` directory.

## Getting Started
After setting up both environments and installing dependencies, you can start exploring the code and running scripts from the `code/` directory by executin main.py to run the full pipeline or execute `sms_eda_notebook.ipynb` for exploratory data analysis(EDA)
For android application simple build and run should work fine.

---
For any issues, please refer to the documentation in `workflow.md` or contact me at 2023AA05482@wilp.bits-pilani.ac.in
