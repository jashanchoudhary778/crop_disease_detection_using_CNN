# 🌿 Crop Disease Detection — ML Project

A deep learning model to detect plant leaf diseases using **Transfer Learning (MobileNetV2)** trained on the **PlantVillage** dataset.

## 📁 Project Structure

```
crop_disease_detection/
├── crop_disease_detection.py     # Full training pipeline (Python script)
├── Crop_Disease_Detection.ipynb  # Step-by-step Jupyter Notebook
├── requirements.txt
└── README.md
```

## 🗂 Dataset

**PlantVillage** — 54,306 images | 38 disease classes | 14 crop types

Download from Kaggle:
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d ./data
```

Expected structure:
```
data/PlantVillage/
    Apple___Apple_scab/
    Apple___Black_rot/
    Tomato___Late_blight/
    Corn___Common_rust/
    ...
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python crop_disease_detection.py
```

Or open the notebook:
```bash
jupyter notebook Crop_Disease_Detection.ipynb
```

## 🧠 Model Architecture

```
ImageNet pretrained MobileNetV2
    └── Frozen backbone (feature extractor)
    └── Custom head:
            Dropout(0.3)
            Linear(1280 → 256) + ReLU
            Linear(256 → 38 classes)
```

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Val Accuracy | ~95% (GPU) |
| Test Accuracy | ~93–95% |
| Training time | ~20 min (GPU) / ~2 hrs (CPU) |

## 🔧 Config Options

Edit `CONFIG` dict in the script:

| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | `./data/PlantVillage` | Dataset path |
| `epochs` | 10 | Training epochs |
| `batch_size` | 32 | Batch size |
| `lr` | 1e-4 | Learning rate |
| `img_size` | 224 | Input image size |

## 📦 Requirements

```
torch>=2.0
torchvision>=0.15
scikit-learn
matplotlib
seaborn
pillow
```

## 🔮 Future Improvements

- Unfreeze last layers for full fine-tuning
- Try EfficientNet-B0 / ResNet50
- Add Grad-CAM visualizations
- Deploy with FastAPI + Streamlit
- Real-time webcam inference

---
**GitHub**: [jashanchoudhary778](https://github.com/jashanchoudhary778)
