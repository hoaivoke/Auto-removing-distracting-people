# Auto Removing Distracting People from Images

Developed by: **Phạm Quốc Bình**, & **Võ Kế Hoài**

---

## 📌 Project Overview
In many photographs, background figures unintentionally degrade image composition or draw focus away from the intended subjects. The goal of this project is to create an automated end-to-end pipeline capable of identifying the main person/subjects in an image, isolating and removing all distracting background individuals, and seamlessly reconstructing the missing background using advanced AI inpainting models.


### Key Objectives:
1. **Target Identification:** Intelligently identify and lock onto the main subject(s) based on spatial cues (centering) and model confidence levels.
2. **Distraction Filtering:** Detect and segment tiny, peripheral, or lower-confidence background figures.
3. **Natural Restoration:** Synthesize missing pixel areas organically using Stable Diffusion inpainting.

---

## ⚙️ Proposed Solution & Workflow
The system utilizes a modern deep learning pipeline split into detection/segmentation stages followed by an inpainting execution block:

1. **Input Stage:** An image containing multiple people is passed to the system.
2. **Fine-tuning & Prediction Stage:**
   - Evaluated models include fine-tuned **YOLOv8-seg**, **Detectron2**, and **PaliGemma** trained/validated using subsets of the **COCO 2017 Dataset**.
   - Zero-shot comparisons are tested against **Grounding DINO** combined with **SAM (Segment Anything Model)**.
3. **Filtering Mask Generation:** The pipeline filters out the main subjects and extracts a binary removal mask targeting only the distracting elements.
4. **Inpainting:** The original image and the removal mask are fed into a **Stable Diffusion Inpainting** model to output a pristine, distraction-free final photograph.

---

## 📊 Evaluation & Models Tested
The repository incorporates notebooks and scripts benchmarking several architectures using **Mask AP (Average Precision)** scores:

* **YOLOv8 (Detection + Segmentation):** Optimized using `yolov8n-seg.pt` for efficient edge-cases and bounding box accuracies categorized by object sizes (Small, Medium, Large).
* **Grounding DINO + SAM:** Offers superb open-vocabulary language-grounded detection paired with precise pixel-level mask segmentation.

---

## 📁 Repository Structure
To maintain a production-ready codebase, the directory structure should look like this:

```text
├── data/
│   ├── coco_person.yaml       # Configuration file for tracking 'person' class paths
│   ├── train/                 # Extracted COCO 2017 training subsets (images/labels)
│   └── val/                   # Extracted COCO 2017 validation subsets
├── notebooks/
│   └── auto-removing-distracting-people.ipynb  # Comprehensive Kaggle/Colab training notebook
├── models/
│   └── yolov8n-seg.pt         # Pretrained or fine-tuned weights
├── src/
│   ├── detect.py              # Scripts managing Grounding DINO/YOLO inference
│   ├── filter.py              # Filtering rules (center proximity, size metrics)
│   └── inpaint.py             # Stable Diffusion pipeline configuration
├── requirements.txt           # Main pipeline dependencies
└── README.md                  # Project documentation
