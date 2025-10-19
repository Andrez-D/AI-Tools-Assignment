
# AI Tools Assignment - Mastering the AI Toolkit 🛠️🧠
## Report
[Download the project documentation](C:\Users\rutoa\Downloads\Documents\# AI Tools Assignment Report.pdf)

## Project Structure
```
ai-tools-assignment/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── theory/
│   └── theoretical_answers.md
│
├── task1_scikit_learn/
│   ├── iris_classification.ipynb
│   └── iris_classification.py
│
├── task2_deep_learning/
│   ├── mnist_cnn_tensorflow.ipynb
│   ├── mnist_cnn_pytorch.ipynb
│   └── model_predictions.png
│
├── task3_nlp/
│   ├── spacy_ner_sentiment.ipynb
│   └── spacy_analysis.py
│
├── ethics/
│   └── ethical_analysis.md
│
├── bonus/
│   └── streamlit_app.py
│
└── report/
    └── AI_Tools_Assignment_Report.pdf
```

## Prerequisites

### System Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Git
- Visual Studio Code
- 4GB RAM minimum (8GB recommended)
- Internet connection for downloading datasets

### Required Software
1. **Python**: [Download Python](https://www.python.org/downloads/)
2. **Git**: [Download Git](https://git-scm.com/downloads)
3. **VS Code**: [Download VS Code](https://code.visualstudio.com/)

## Setup Instructions

### Step 1: Clone or Create Repository

#### Option A: Start Fresh
```bash
# Create new directory
mkdir ai-tools-assignment
cd ai-tools-assignment

# Initialize git
git init

# Create project structure
mkdir theory task1_scikit_learn task2_deep_learning task3_nlp ethics bonus report
```

#### Option B: Clone from GitHub (after initial push)
```bash
git clone https://github.com/YOUR_USERNAME/ai-tools-assignment.git
cd ai-tools-assignment
```

### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter issues, install individually:
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow
pip install torch torchvision
pip install spacy nltk
pip install jupyter notebook
pip install streamlit flask

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Step 4: Open in VS Code

```bash
# Open the project in VS Code
code .
```

### Step 5: Configure VS Code

1. Install Python extension (ms-python.python)
2. Install Jupyter extension (ms-toolsai.jupyter)
3. Select Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose your venv

### Step 6: Download Datasets

The code will automatically download most datasets. For Amazon Reviews:
```bash
# Create data directory
mkdir data
cd data

# Download from Kaggle (requires Kaggle API setup)
# Or manually download from the provided links
```

## Running the Project

### Task 1: Scikit-learn (Iris Classification)

```bash
# Navigate to task directory
cd task1_scikit_learn

# Run Jupyter notebook
jupyter notebook iris_classification.ipynb

# Or run Python script
python iris_classification.py
```

### Task 2: Deep Learning (MNIST CNN)

```bash
cd task2_deep_learning

# For TensorFlow version
jupyter notebook mnist_cnn_tensorflow.ipynb

# For PyTorch version
jupyter notebook mnist_cnn_pytorch.ipynb
```

### Task 3: NLP (spaCy NER & Sentiment)

```bash
cd task3_nlp

# Run Jupyter notebook
jupyter notebook spacy_ner_sentiment.ipynb

# Or run Python script
python spacy_analysis.py
```

### Bonus: Streamlit Web App

```bash
cd bonus

# Run Streamlit app
streamlit run streamlit_app.py

# App will open in browser at http://localhost:8501
```

## Pushing to GitHub

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click "New Repository"
3. Name: `ai-tools-assignment`
4. Don't initialize with README (we have one)
5. Click "Create Repository"

### Step 2: Push Your Code

```bash
# Add all files
git add .

# Commit changes
git commit -m "Initial commit: AI Tools Assignment complete solution"

# Add remote origin (replace with your username)
git remote add origin https://github.com/YOUR_USERNAME/ai-tools-assignment.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Verify on GitHub

- Visit your repository URL
- Ensure all files are uploaded
- Check that README displays correctly

## Generating the Report PDF

### Using Jupyter Notebooks

1. Open each notebook in Jupyter
2. Run all cells: `Cell` → `Run All`
3. Export as PDF: `File` → `Download as` → `PDF via LaTeX`
4. Combine all exports into one report

### Using Markdown to PDF

```bash
# Install pandoc
# Windows: Download from https://pandoc.org/installing.html
# macOS: brew install pandoc
# Linux: sudo apt-get install pandoc

# Generate PDF from markdown
pandoc theory/theoretical_answers.md ethics/ethical_analysis.md -o report/AI_Tools_Assignment_Report.pdf
```

## Creating the Video Presentation

### Tools
- **Screen Recording**: OBS Studio, Loom, or Zoom
- **Video Editing**: DaVinci Resolve (free), iMovie, or Windows Video Editor

### Script Outline (3 minutes)
1. **Introduction (30s)**: Team intro, project overview
2. **Task 1 Demo (45s)**: Show Iris classification results
3. **Task 2 Demo (45s)**: Show MNIST CNN predictions
4. **Task 3 Demo (45s)**: Show NER and sentiment analysis
5. **Ethics & Conclusion (15s)**: Key insights and takeaways

### Recording Tips
- Test audio before recording
- Use clear, simple slides
- Show code running live
- Keep energy high and pace brisk

## Troubleshooting

### Common Issues

**ImportError: No module named 'tensorflow'**
```bash
pip install tensorflow
```

**CUDA/GPU Issues**
```python
# Check if GPU is available
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# If no GPU, code will run on CPU (slower but works)
```

**spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

**Jupyter Kernel Issues**
```bash
python -m ipykernel install --user --name=venv
```

**Git Push Authentication**
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys

## Evaluation Criteria

- ✅ **Theoretical Accuracy (30%)**: Complete answers in `theory/`
- ✅ **Code Functionality (40%)**: All tasks run without errors
- ✅ **Ethical Analysis (15%)**: Thoughtful bias analysis
- ✅ **Creativity & Presentation (15%)**: Clear video and documentation

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [spaCy Documentation](https://spacy.io/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## License

MIT License - Feel free to use this code for educational purposes.

## Acknowledgments

- Course Instructor and TAs
- Open-source AI community
- Dataset providers on Kaggle

---
