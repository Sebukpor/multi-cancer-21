# 🧬 Browser-Based Multi-Cancer Classification Framework Using Depthwise Separable Convolutions  

![TensorFlow.js](https://img.shields.io/badge/Made%20with-TensorFlow.js-orange?logo=tensorflow)    ![License](https://img.shields.io/badge/License-MIT-blue.svg)  ![Status](https://img.shields.io/badge/Demo-Live-success?logo=huggingface)   ![Platform](https://img.shields.io/badge/Platform-Browser--Based-green)

---

## 📌 Overview

This repository presents the official implementation of the work titled **"Browser-Based Multi-Cancer Classification Framework Using Depthwise Separable Convolutions for Precision Diagnostics"**, currently **under peer review**.  

We introduce a **lightweight, privacy-preserving, and browser-accessible** deep learning system capable of classifying **26 distinct cancer types** from diverse medical imaging modalities—including MRI, histopathology, cytology, and CT scans—using a fine-tuned **Xception architecture** with **depthwise separable convolutions**. Inference runs **entirely client-side** via **TensorFlow.js**, enabling deployment in low-resource and privacy-sensitive settings without requiring cloud infrastructure.

---

## 🧪 Reproducibility Package

This repository provides **complete code**, **training scripts**, **conversion utilities**, and **documentation** to reproduce:

- The **26-class multi-cancer classification model** (99.85% Top-1 accuracy)
- **Client-side browser deployment** using TensorFlow.js
- **Grad-CAM interpretability visualizations**
- **Training pipeline** with data augmentation and hyperparameter settings


---

## ⚙️ Training Setup & Dependencies

### **Datasets**
The model was trained on a **curated composite dataset** of **130,000+ images** from 8 public sources:

| Cancer Type         | Dataset Source (Kaggle/Figshare)                                                                 |
|---------------------|--------------------------------------------------------------------------------------------------|
| Brain Tumor         | [Figshare – Cheng (2017)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)      |
| Acute Lymphoblastic Leukemia | [Kaggle – ALL-IDB](https://www.kaggle.com/datasets/mehradaria/leukemia)                   |
| Breast Cancer       | [BreakHis – Spanhol et al.](https://www.kaggle.com/datasets/anaselmasry/breast-cancer-dataset)   |
| Cervical Cancer     | [SIPaKMeD – Plissiti et al.](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed) |
| Kidney CT           | [CT-Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) |
| Lung & Colon        | [LC25000 – Borkowski et al.](https://www.kaggle.com/datasets/biplobdey/lung-and-colon-cancer)    |
| Lymphoma            | [Malignant Lymphoma](https://www.kaggle.com/datasets/andrewmvd/malignant-lymphoma-classification)|
| Oral Cancer         | [Oral Histopathology](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset)               |

> 🔗 **Preprocessed, balanced dataset (train/val/test)**:  
> [https://www.kaggle.com/datasets/maestroalert/cancer](https://www.kaggle.com/datasets/maestroalert/cancer)

### **Class List (26)**
```python
Folder Names
[
 'all_benign', 'all_early', 'all_pre', 'all_pro',
 'brain_glioma', 'brain_menin', 'brain_tumor',
 'breast_benign', 'breast_malignant',
 'cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi',
 'colon_aca', 'colon_bnt',
 'kidney_normal', 'kidney_tumor',
 'lung_aca', 'lung_bnt', 'lung_scc',
 'lymph_cll', 'lymph_fl', 'lymph_mcl',
 'oral_normal', 'oral_scc'
]
Actual Names
[ 'Acute Lymphoblastic Leukemia Benign', 'Acute Lymphoblastic Leukemia Early',
    'Acute Lymphoblastic Leukemia Pre', 'Acute Lymphoblastic Leukemia Pro',
    'Brain Glioma', 'Brain Meningioma', 'Brain Tumor',
    'Breast Benign', 'Breast Malignant',
    'Cervix Dyskeratotic', 'Cervix Koilocytotic', 'Cervix Metaplastic',
    'Cervix Parabasal', 'Cervix Superficial Intermediate',
    'Colon Adenocarcinoma', 'Colon Benign Tissue',
    'Kidney Normal', 'Kidney Tumor',
    'Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma',
    'Chronic Lymphocytic Leukemia', 'Follicular Lymphoma', 'Mantle Cell Lymphoma',
    'Oral Normal', 'Oral Squamous Cell Carcinoma'
]

```

### **Environment & Dependencies**
- Python ≥ 3.8
- TensorFlow ≥ 2.13
- OpenCV, NumPy, Pandas, Matplotlib, Seaborn
- `tensorflowjs` (for browser conversion)

Install via:
```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn tensorflowjs
```

---

## 🛠️ Training Protocol (from `xception-model.ipynb`)

| Parameter               | Value                         |
|------------------------|-------------------------------|
| Base Model             | Xception (ImageNet weights)   |
| Input Size             | 224 × 224 × 3                 |
| Batch Size             | 32                            |
| Epochs                 | 21                            |
| Optimizer              | Adam (lr = 1e⁻⁴)              |
| Loss                   | Categorical Cross-Entropy     |
| Augmentation           | Rotation (45°), zoom (0.2), horizontal/vertical flip, shift |
| Regularization         | Dropout (0.1–0.4), L2 weight decay |
| Fine-tuned Layers      | Last 50 layers of Xception    |
| Callbacks              | EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.2, patience=5) |

> 💡 The model is **class-balanced**: 4,000 training images per class, 500 validation/test per class.

---

## 📈 Performance Metrics (Test Set)

| Metric                     | Value     |
|---------------------------|-----------|
| **Top-1 Accuracy**        | **99.85%**|
| **Top-5 Accuracy**        | **100.00%**|
| **Macro Precision**       | 1.00      |
| **Macro Recall**          | 1.00      |
| **Macro F1-Score**        | 1.00      |

> 📊 See `training_validation_metrics.png` and `confusion_matrix.png`.

---

## 🌐 Browser Deployment (TensorFlow.js)

The trained model is converted for **client-side inference**:

```bash
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --signature_name=serving_default \
  --saved_model_tags=serve \
  saved_model/Pneumonia/ \
  tfjs_model/
```

✅ **Key advantages**:
- **Zero data upload** — all processing in-browser
- **No server required** — works offline after load
- **GDPR/HIPAA-compliant** by design
- **Accessible globally** — runs on any modern browser

🔗 **Live Demo**:  
[https://huggingface.co/spaces/Sebukpor/multi-cancer-gradcam](https://huggingface.co/spaces/Sebukpor/multi-cancer-gradcam)

🔗 **Web App**:  
[https://sebukpor.github.io/multi-cancer-classification/](https://sebukpor.github.io/multi-cancer-classification/)

---

## 🔍 Interpretability: Grad-CAM

The model includes **Grad-CAM support** (implemented in the Hugging Face demo) to generate **heatmaps** highlighting diagnostically relevant regions. This builds clinical trust by ensuring predictions are grounded in **histopathological features**, not artifacts.

Example:  
- Lung adenocarcinoma → heatmap focuses on **atypical glandular cells**  
- Breast malignancy → activation on **infiltrating malignant sheets**

---

## 🧭 Limitations & Future Work

While the current model achieves state-of-the-art results, the following **research directions** are recommended:

1. **External Clinical Validation**  
   → Test on **multi-institutional, real-world data** with diverse staining protocols and scanners.

2. **Federated Learning**  
   → Train across decentralized hospitals **without sharing raw images**.

3. **Multi-Modal Integration**  
   → Fuse imaging with **genomic, clinical, or lab data** for holistic diagnosis.

4. **Rare Cancer Expansion**  
   → Include underrepresented cancers (e.g., sarcomas, pediatric tumors).

5. **Prospective Clinical Trials**  
   → Validate impact on **diagnostic accuracy, time-to-diagnosis, and clinician workflow**.

> 📬 **We welcome collaborations** for further clinical validation and dataset sharing.

---

## 📄 Citation & Paper Status

**Preprint**:  
[https://www.preprints.org/manuscript/202510.1612](https://www.preprints.org/manuscript/202510.1612)

If you use this code or model in your research, please cite:

```bibtex
@article{sebukpor2025browser,
  title={Browser-Based Multi-Cancer Classification Framework Using Depthwise Separable Convolutions for Precision Diagnostics},
  author={Sebukpor, Divine and Odezuligbo, Ikenna and Nagey, Maimuna and Chukwuka, Michael and Akinsuyi, Oluwamayowa},
  journal={Preprints},
  year={2025},
  doi={10.20944/preprints202510.1612.v1}
}
```

---

## 🤝 Contact & Contribution

**Corresponding Author**:  
Oluwamayowa Akinsuyi  
📧 [divinesebukpor@gmail.com](mailto:divinesebukpor@gmail.com)

**We welcome**:
- Bug reports
- Dataset contributions
- Clinical validation partnerships
- Pull requests for enhancements

---

## 📜 License

MIT License – see [LICENSE](LICENSE) for details.

> 🌍 **Our mission**: Democratize AI-powered cancer diagnostics for **global health equity**.

--- 
