# ATS Resume Analyzer - Setup & Prerequisites

## 1. Prerequisites
Before you begin, ensure you have the following installed on your machine:
- **Python 3.10 or higher** (Verify by running `python --version` in your terminal)
- **Windows OS** (for using `setup.bat`) or **Mac/Linux** (for using `setup.sh`)

---

## 2. Initial Setup (One-Time)
We have provided automated setup scripts to make installation effortless for all teammates. These scripts will create an isolated virtual environment and install all necessary AI dependencies without affecting your global system.

### For Windows Users:
1. Open the project folder in File Explorer.
2. Double-click the `setup.bat` file.
3. Wait for the terminal window to display "Setup Complete!".

### For Mac/Linux Users:
1. Open your terminal and navigate to the project directory.
2. Make the script executable by running: 
   ```bash
   chmod +x setup.sh
   ```
3. Execute the setup script: 
   ```bash
   ./setup.sh
   ```

---

## 3. How to Run & Start Testing
Once the setup is complete, you can run the candidate ranking pipeline.

### Step 1: Activate the Environment (Mandatory)
Every time you open a new terminal, you must activate the environment:
- **Windows:** `call .venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### Step 2: Run the Default Pipeline
To test the pipeline with the provided sample data (100k candidates), simply run:
```bash
python run_submission.py
```

### Step 3: Test with Custom Data (Optional)
If you want to use your own hidden test datasets without replacing the default files, you can pass them as arguments:
```bash
python run_submission.py --candidates "path/to/custom.jsonl" --jd "path/to/custom.docx"
```

### Step 4: Check the Output
The pipeline will execute entirely locally (zero-network) and generate a `submission.csv` file in the root directory containing the top 100 ranked candidates.
