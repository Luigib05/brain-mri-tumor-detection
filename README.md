# brain-mri-tumor-detection
This project explores brain tumor detection using transfer learning on MRI images. We compare the performance of three CNN architectures: VGG16, EfficientNetB0 and LeNet

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
├── brain_MRI_images_brain_tumor_detection_VGG16.ipynb   # VGG16 model
├── brain_MRI_images_brain_tumor_detection_EfficientNet.ipynb   # EfficientNetB0 model
|-- brain_MRI_images_brain_tumor_detection_Lenet.ipynb  #Lenet model
├── README.md
├── requirements.txt

## Model Comparison

Three different convolutional neural network architectures were evaluated for binary classification of brain MRI images into "no tumor" and "tumor" categories:

| Model            | Accuracy | AUC   | Key Observations |
|------------------|----------|-------|------------------|
| **VGG16**        | 0.6667   | 0.7532 | Very high recall for "tumor" (1.00), but fails to detect "no tumor" cases (recall = 0.15). Strong bias toward the majority class. |
| **EfficientNetB0** | 0.6078 | 0.4863 | Weak overall performance. Fails to distinguish between classes (AUC near 0.5). Grad-CAM failed to produce informative maps. |
| **LeNet**        | 0.7843   | 0.8700 | Best overall result. Balanced precision and recall for both classes. Grad-CAM partially successful. |

---

## Project Limitations

1. **Small dataset size**: The limited number of samples affects the generalization ability, especially for complex models like VGG16 and EfficientNetB0.
2. **Class imbalance**: More images labeled as "tumor" lead to training bias and poor performance on "no tumor" detection.
3. **Overfitting and inefficiency**:
   - VGG16 tends to always predict "tumor".
   - EfficientNetB0 failed to learn meaningful patterns.
   - LeNet showed better generalization likely due to its simplicity.
4. **Grad-CAM limitations**: Grad-CAM was only partially functional on LeNet. For EfficientNetB0, activation maps were not meaningful, and the Sequential architecture caused issues in interpretability callbacks.

---

## Future Improvements

- **Increase dataset size**, especially for "no tumor" cases, to reduce class imbalance and improve generalization.
- **Apply data augmentation** (rotation, zoom, brightness shifts) to improve model robustness.
- **Explore simpler CNNs** (e.g., LeNet, AlexNet, or custom CNNs) better suited for small datasets.
- **Implement class balancing strategies**:
  - Use of `class_weight` during training.
  - Oversampling the minority class.
- **Convert models to Functional API** to enhance compatibility with interpretability tools like Grad-CAM.
- **Incorporate clinical metrics** such as inference time and interpretability to align with real-world medical use cases.

---
## Conclusion

This project explored the classification of brain MRI images using three convolutional neural network architectures: VGG16, EfficientNetB0, and LeNet. Despite the high potential of pre-trained and deeper models, their performance was suboptimal due to the small and imbalanced dataset. In contrast, LeNet — a simpler architecture — delivered the best results in terms of accuracy, AUC, and class balance.

These findings emphasize the importance of matching model complexity to dataset size and quality. For practical deployments in medical imaging, especially in low-data scenarios, simplicity and interpretability often outperform depth and sophistication.

While Grad-CAM interpretability tools showed mixed results across models, future work should focus on improving data quality, expanding dataset size, and ensuring compatibility between architectures and visualization methods.

This project provides a reproducible baseline and highlights key considerations for future improvements in deep learning applications to medical imaging.


-----------------
## Author
Project developed by Luigib05 as a transfer learning (TL) exercise applied to AI-assisted medical diagnosis.




