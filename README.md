# brain-mri-tumor-detection
This project explores brain tumor detection using transfer learning on MRI images. We compare the performance of two CNN architectures: VGG16 and EfficientNetB0

# Brain MRI Tumor Detection

This project uses *Deep Learning* (Transfer Learning) models to classify brain magnetic resonance images (MRIs) into two categories: **with tumor** (`yes`) and **without tumor** (`no`). A complete binary classification pipeline based on computer vision is implemented with interpretability using Grad-CAM.

---

## Dataset

**Name:** Brain MRI Images for Brain Tumor Detection  
**Source:** [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
**Content:**  
- 98 images labeled as `no` (without tumor)  
- 155 imáges labeled as `yes` (with tumor)  
- Format: `.jpg`  
- Classes separated by folders

---

## Objective

Develop a medical image classifier to help detect brain tumors from MRIs. It also aims to provide transparency through Grad-CAM activation maps.

---

## Technologies

- Python 3
- Google Colab
- TensorFlow / Keras
- OpenCV / Matplotlib / Seaborn
- scikit-learn

---

## Project structure

```bash
brain-mri-tumor-detection/
│
├── brain_MRI_images_brain_tumor_detection_VGG16.ipynb   # Modelo VGG16
├── brain_MRI_images_brain_tumor_detection_EfficientNet.ipynb   # (Futuro)
├── README.md
├── requirements.txt

-----------------
## Author
Project developed by Luigib05 as a transfer learning (TL) exercise applied to AI-assisted medical diagnosis.




