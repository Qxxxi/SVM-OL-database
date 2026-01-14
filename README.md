# Multimodal Emotion Recognition in Online Learning

A multimodal emotion recognition framework for online learning scenarios, combining visual and textual modalities to classify student emotions into five categories: **Bored**, **Happy**, **Interested**, **Tired**, and **Confused**.

## Overview

This project implements:
- **Single-modality models**: Visual (V) and Textual (T) emotion recognition
- **Bimodal fusion**: V+T fusion with various fusion strategies
- **Trimodal fusion**: CLIP + OpenFace/Face + BERT fusion
- **3-fold cross-validation**: Person-independent data splitting

## Project Structure

```
SVM-OL_code/
├── README.md                           # This file
├── utilz.py                            # Utility functions
├── generate_3fold_person_split.py      # Generate 3-fold person-independent splits
│
├── T/                                  # Textual Modality
│   ├── text_processed.ipynb            # Text feature extraction
│   └── text_emotion_recognition_reorganized.ipynb  # Text emotion models
│
├── V/                                  # Visual Modality
│   ├── visual_processed.ipynb          # Visual feature extraction
│   └── visual_emotion_recognition.ipynb # Visual emotion models
│
├── V+T/                                # Multimodal Fusion
│   ├── multimodal_fusion_comparison.ipynb  # Bimodal fusion comparison
│   ├── trimodal_face_fusion.ipynb      # CLIP + Face + BERT fusion
│   └── trimodal_OFfts_fusion.ipynb     # CLIP + OpenFace + BERT fusion
│
│── calculate_average_crossfold/        # Cross-fold Average Calculation
│    ├── calculate_t_or_v_average_crossfold.ipynb
│    ├── calculate_trimodal_face_crossfold.ipynb
│    ├── calculate_trimodal_offt_crossfold.ipynb
│    └── calculate_fusion_strategies_crossfold.ipynb

##  Other Documents

checkpoints
│── data/                                   # Feature files
│     ├── fold{0,1,2}_labels.pkl              # Labels for each fold
│     ├── fold{0,1,2}_textual_wav2vec.pkl     # Word2Vec text features
│     ├── fold{0,1,2}_textual_bert.pkl        # BERT text features
│     ├── fold{0,1,2}_textual_baidu.pkl       # Baidu Baike word vectors
│     ├── fold{0,1,2}_visual_clip.pkl         # CLIP visual features
│     ├── fold{0,1,2}_visual_face.pkl         # Face image features (MediaPipe)
│     └── fold{0,1,2}_visual_OFfts.pkl        # OpenFace AU features
│
│──result/                                 # Training results
│     ├── resm/                               # Multimodal results
│     └── resv_or_t/                          # Single modality results
│
│──sample/                                 # Sample data
│     ├── [OpenFace processed samples]
│     └── [Label files]

## Requirements

```bash
# Core dependencies
tensorflow>=2.10.0
keras
numpy
pandas
scikit-learn
matplotlib
seaborn

# Text processing
transformers      # For BERT
gensim           # For Word2Vec
jieba            # Chinese word segmentation

# Visual processing
opencv-python
mediapipe        # Face detection
clip             # OpenAI CLIP
torch            # For CLIP

# Optional (for OpenFace features)
# OpenFace 2.2.0 - requires separate installation
```

## Data Preparation

### 1. Generate 3-Fold Person-Independent Splits

```bash
python generate_3fold_person_split.py
```

**Input**: `data2025.csv` with columns:
- `index`: Sample ID
- `path`: Video file path
- `nlp`: Transcribed text
- `classes`: Emotion label (无聊/快乐/感兴趣/疲倦/困惑)
- `mode`: train/test split

**Output**:
- `fold0_train.csv`, `fold1_train.csv`, `fold2_train.csv`
- Each fold uses different people for testing (person-independent)

### 2. Extract Text Features

Run `T/text_processed.ipynb` to extract:
- **Word2Vec**: Self-trained on dataset, 100-dim
- **BERT**: bert-base-chinese, 768-dim
- **Baidu Baike**: Pre-trained Chinese word vectors, 300-dim

### 3. Extract Visual Features

Run `V/visual_processed.ipynb` to extract:
- **CLIP**: OpenAI CLIP ViT-B/32, 512-dim per frame
- **Face**: MediaPipe face detection + CNN features
- **OpenFace**: Facial Action Units (AUs) + landmarks, 709-dim

## Model Architectures

### Text Models (`T/text_emotion_recognition_reorganized.ipynb`)

| Model | Architecture | Features |
|-------|-------------|----------|
| TextCNN | Multi-scale Conv1D (3,4,5) + GlobalMaxPool | Word2Vec/BERT/Baidu |
| LSTM | BiLSTM + GlobalMaxPool | Word2Vec/BERT/Baidu |
| Att+BiLSTM | Self-Attention + BiLSTM + GlobalMaxPool | Word2Vec/BERT/Baidu |

### Visual Models (`V/visual_emotion_recognition.ipynb`)

| Model | Architecture | Features |
|-------|-------------|----------|
| CNN+LSTM | Conv1D + LSTM | CLIP/OpenFace |
| 3D-CNN | TimeDistributed CNN + LSTM | Face images |

### Fusion Methods (`V+T/multimodal_fusion_comparison.ipynb`)

| Method | Description | Reference |
|--------|-------------|-----------|
| Concat | Simple concatenation (baseline) | - |
| LMF | Low-rank Multimodal Fusion | ACL 2018 |
| Gated | Learnable gating mechanism | - |
| TFN | Tensor Fusion Network | EMNLP 2017 |
| MulT | Multimodal Transformer | ACL 2019 |

### Trimodal Fusion Strategies (`V+T/trimodal_*.ipynb`)

| Strategy | Description |
|----------|-------------|
| Concat | Simple concatenation |
| Weighted Sum | Learnable weighted sum |
| Gated | Three-way gating |
| Hierarchical | Visual-first, then cross-modal |
| Bilinear | Bilinear interaction |
| Cross-Attention | Cross-modal attention |
| Tensor | Tensor product fusion |

## Usage

### Training Single-Modality Models

```python
# Text emotion recognition
# Open T/text_emotion_recognition_reorganized.ipynb
# Modify fold number in data loading section
# Run all cells

# Visual emotion recognition
# Open V/visual_emotion_recognition.ipynb
# Modify fold number in data loading section
# Run all cells
```

### Training Multimodal Fusion Models

```python
# Bimodal fusion (CLIP + BERT)
# Open V+T/multimodal_fusion_comparison.ipynb
# Run all cells to compare 5 fusion methods

# Trimodal fusion (CLIP + OpenFace + BERT)
# Open V+T/trimodal_OFfts_fusion.ipynb
# Modify FOLD variable (0, 1, or 2)
# Run all cells
```

### Calculating Cross-Fold Average Results

```python
# Open calculate_average_crossfold/calculate_t_or_v_average_crossfold.ipynb
# Configure model paths for each fold
# Run all cells to get mean ± std metrics
```

## Key Functions (`utilz.py`)

```python
# Save/Load pickle features
save_features(data, file_name)
data = load_features(file_name)

# Load CSV data
ids, paths, nlps, classes, modes = load_data('fold0_train.csv')

# Get OpenFace feature column names
feature_columns = OP_para()  # Returns 709 feature names
```

## Emotion Classes

| Class ID | Chinese | English |
|----------|---------|---------|
| 0 | 无聊 | Bored |
| 1 | 快乐 | Happy |
| 2 | 感兴趣 | Interested |
| 3 | 疲倦 | Tired |
| 4 | 困惑 | Confused |

## Results Format

Results are saved in multiple formats:
- `.tf` - TensorFlow SavedModel format
- `.csv` - Metrics summary
- `.xlsx` - Detailed Excel reports
- `.png` - Visualization plots

Metrics include:
- Accuracy (per fold and average)
- Precision, Recall, F1-score (per class)
- Macro/Weighted averages
- Confusion matrices

## Tips

1. **GPU Memory**: For Face image models, use smaller batch size (8) due to image data size
2. **Class Imbalance**: All models use computed class weights
3. **Reproducibility**: Set random seed (42) in data splitting
4. **Early Stopping**: All models use patience=10-15 epochs

## Acknowledgments

- OpenFace 2.2.0 for facial action unit extraction
- OpenAI CLIP for visual encoding
- Hugging Face Transformers for BERT
- Baidu Baike word vectors
