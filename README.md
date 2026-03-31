# Emotion Detection from Face (BYOP)

## Problem
People often struggle to express their mood directly. A lightweight facial emotion detection system can help applications understand user mood and respond better.

## Solution
This project detects three emotions from face images:
- happy
- sad
- angry

The pipeline:
1. Detect face from image/webcam frame
2. Extract features from face ROI
3. Classify emotion using either:
   - `KNN` (K-Nearest Neighbors)
   - `ANN` (Artificial Neural Network via `MLPClassifier`)

## Syllabus Mapping
- Feature Extraction: grayscale normalization + HOG (Histogram of Oriented Gradients)
- Classification: KNN and ANN models

## Project Structure
```
cvproject/
|-- data/
|   |-- raw/
|   |   |-- happy/
|   |   |-- sad/
|   |   `-- angry/
|   `-- processed/
|-- models/
|   `-- .gitkeep
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- features.py
|   |-- train.py
|   |-- predict.py
|   `-- realtime.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Dataset Format
Place your images here:

```
data/raw/happy/*.jpg
data/raw/sad/*.jpg
data/raw/angry/*.jpg
```

Download source used for this project:
- FER2013 Kaggle dataset: https://www.kaggle.com/datasets/msambare/fer2013

Important:
- The raw dataset is not committed to GitHub because of repository size and file-size limits.
- Keep only folder placeholders in the repository and place dataset files locally in data/raw/happy, data/raw/sad, and data/raw/angry.
- Do not upload dataset zip files inside the repository history.

If your download has labels instead of class folders, map labels as:
- `0 -> angry`
- `3 -> happy`
- `4 -> sad`

Tips:
- Use clear face images
- Keep classes balanced (roughly similar number of images per emotion)
- You can mix `.jpg`, `.jpeg`, `.png`

## Setup
### 1. Create virtual environment
Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

Alternative (without activation), using full interpreter path:
```powershell
c:/Users/shria/OneDrive/Desktop/cvproject/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## How to Run
### 1. Train model (KNN)
```powershell
python main.py train --model knn
```

### 2. Train model (ANN)
```powershell
python main.py train --model ann
```

Alternative (full interpreter path):
```powershell
c:/Users/shria/OneDrive/Desktop/cvproject/.venv/Scripts/python.exe main.py train --model knn
c:/Users/shria/OneDrive/Desktop/cvproject/.venv/Scripts/python.exe main.py train --model ann
```

By default, trained model is saved to:
`models/emotion_model.joblib`

### 3. Predict emotion for one image
```powershell
python main.py predict --image path\to\test.jpg
```

Example:
```powershell
python main.py predict --image data\raw\happy\Training_10019449.jpg
```

### 4. Real-time webcam detection
```powershell
python main.py realtime
```

Press `q` to quit webcam window.

## Example Commands
```powershell
python main.py train --model ann --test-size 0.2 --random-state 42
python main.py predict --image data\raw\happy\sample1.jpg
python main.py realtime --camera 0
```

## Quick Verification (Recommended)
Run these commands in order after placing data:

```powershell
python main.py train --model knn
python main.py train --model ann
python main.py predict --image data\raw\happy\Training_10019449.jpg
python main.py realtime
```

You should see:
- printed accuracy and classification report after each training run
- `models/emotion_model.joblib` created/updated
- predicted emotion + confidence for single image
- webcam window with face box and emotion label overlay

## GitHub Upload Policy (Important)
For submission, upload source code and documentation only.

Do upload:
- source code files
- requirements.txt
- README.md
- folder placeholders (.gitkeep)

Do not upload:
- raw dataset files in data/raw
- external downloaded data in data/external
- trained model binaries (unless explicitly asked by evaluator)
- zipped dataset archives

If dataset files were already tracked by git, run once:

```powershell
git rm -r --cached data/raw
git rm -r --cached data/external
git add .
git commit -m "Remove dataset files from git tracking"
git push
```

If you want to share data, use a link in README (Kaggle/Drive/Release), not repository history.

## Expected Output
- Training metrics in terminal (accuracy and class-wise report)
- Saved model artifact in `models/`
- For prediction: emotion label with confidence score
- For real-time: face box + emotion overlay on webcam feed

## Troubleshooting
- `No images found`: verify folder names are exactly `happy`, `sad`, `angry`
- `No face detected`: this pipeline now falls back to whole grayscale image, but clearer frontal faces still improve accuracy
- Webcam not opening: try `--camera 1` or close apps using camera
- Low accuracy: increase data size and class balance

## GitHub Submission Checklist
- Push all source files and commit history
- Keep trained model optional (you can regenerate with `train` command)
- Keep dataset out of repository history
- Ensure README explains setup and usage clearly
