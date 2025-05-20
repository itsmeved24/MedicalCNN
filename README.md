# Brain Tumor Detection using CNN

This project implements a Convolutional Neural Network (CNN) for brain tumor detection from medical images. The model is trained to classify brain MRI scans into two categories: with tumor and without tumor.

## Project Structure

```
.
├── cnn_train.py          # Main training script
├── brain_tumor_dataset/  # Dataset directory (not included in repo)
│   ├── yes/             # Images with tumors
│   └── no/              # Images without tumors
└── requirements.txt      # Project dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses a brain tumor dataset with the following structure:
- `brain_tumor_dataset/yes/`: Contains MRI scans with tumors
- `brain_tumor_dataset/no/`: Contains MRI scans without tumors

## Training

To train the model:
```bash
python cnn_train.py
```

The script will:
- Load and preprocess the dataset
- Train the CNN model
- Save the best model as 'cnn_brain_tumor.pth'
- Generate a confusion matrix visualization

## Dependencies

The project requires the following Python packages:
- torch
- torchvision
- PIL (Pillow)
- scikit-learn
- matplotlib
- seaborn

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Model Architecture

The CNN model consists of:
- 3 convolutional layers with ReLU activation and max pooling
- Fully connected layers with dropout for classification
- Input image size: 128x128 pixels
- Output: Binary classification (tumor/no tumor)

## License

[Add your chosen license here] 