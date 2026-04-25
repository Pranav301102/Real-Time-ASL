# Real-Time-ASL
# Real-Time ASL Recognition

Converts American Sign Language gestures captured from a webcam into English text using MediaPipe hand keypoint extraction, a bidirectional LSTM, and LLM post-processing.

---

## How it works

The pipeline runs in four stages:

1. **Keypoint extraction** — MediaPipe Hands extracts 21 (x, y, z) landmarks per hand from each webcam frame. A sliding window of T consecutive frames forms the input sequence fed to the classifier.

2. **Isolated sign recognition** — A bidirectional LSTM classifies each window as a known ASL word (trained on WLASL) or fingerspelled character (trained on Google's ASL Fingerspelling dataset).

3. **Continuous sign recognition** — Each window spans 3 seconds of frames. A label only enters the output buffer when it appears consistently as the dominant prediction across a 3-second window. A pause is detected when a 3-second window produces no confident prediction, at which point the buffer is flushed to the next stage.

4. **LLM post-processing** — The buffered sign labels are sent to an LLM as a plain ordered list. The LLM corrects likely misclassifications and converts ASL gloss order into English word order. Only labels are sent — no keypoints or embeddings.

---

## Project structure

```
asl-recognition/
├── data/
│   ├── wlasl/                  # WLASL video dataset
│   └── fingerspelling/         # Google ASL Fingerspelling dataset
├── notebooks/
│   └── train.ipynb             # Google Colab training notebook
├── src/
│   ├── extract_keypoints.py    # MediaPipe keypoint extraction
│   ├── dataset.py              # PyTorch dataset and dataloader
│   ├── model.py                # Bidirectional LSTM model
│   ├── train.py                # Training loop
│   ├── inference.py            # Real-time webcam inference
│   └── postprocess.py          # LLM post-processing
├── checkpoints/                # Saved model weights
├── requirements.txt
└── README.md
```

---

## Setup

**Prerequisites:** Python 3.9+, a webcam, and a Google account for Colab training.

```bash
git clone https://github.com/your-username/asl-recognition.git
cd asl-recognition
pip install -r requirements.txt
```

**requirements.txt**
```
torch
mediapipe
opencv-python
numpy
openai          # or anthropic, depending on your LLM choice
tqdm
scikit-learn
```

---

## Datasets

**WLASL** — Word-level ASL video clips.
Download from the [WLASL GitHub repository](https://github.com/dxli94/WLASL) and place videos under `data/wlasl/`.

**Google ASL Fingerspelling** — MediaPipe keypoints paired with fingerspelled phrase labels.
Download from [Kaggle](https://www.kaggle.com/competitions/asl-fingerspelling/data) and place under `data/fingerspelling/`.

---

## Training (Google Colab)

Open `notebooks/train.ipynb` in Google Colab. The notebook covers:

- Mounting Google Drive to load data and save checkpoints
- Keypoint extraction from WLASL videos using MediaPipe
- Building the PyTorch dataset and dataloader
- Defining and training the bidirectional LSTM
- Evaluating top-1 and top-5 accuracy
- Saving the best checkpoint to Drive

Make sure to set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU) before running.

After training, download the checkpoint file and place it in the local `checkpoints/` folder for inference.

---

## Inference (local)

Run the real-time webcam demo:

```bash
python src/inference.py --checkpoint checkpoints/lstm_best.pt --labels data/labels.json
```

**What happens:**
- Your webcam opens and MediaPipe begins extracting hand keypoints frame by frame.
- Every 3 seconds, the LSTM classifies the window and checks for a stable prediction.
- Stable labels accumulate in a buffer. A 3-second window with no confident prediction triggers a flush.
- The buffered labels are sent to the LLM, which returns a natural English sentence printed to the terminal.

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to trained model weights |
| `--labels` | required | Path to label index JSON |
| `--window` | 90 | Frames per 3-second window (assumes 30fps) |
| `--confidence` | 0.6 | Minimum prediction probability to count as confident |
| `--camera` | 0 | Camera device index |

---

## Evaluation

To evaluate the trained model on a held-out test split:

```bash
python src/train.py --eval --checkpoint checkpoints/lstm_best.pt
```

Metrics reported: top-1 accuracy, top-5 accuracy, and average inference latency per window.

---

## Datasets and acknowledgements

- [WLASL](https://github.com/dxli94/WLASL) — Li et al., 2020
- [Google ASL Fingerspelling](https://www.kaggle.com/competitions/asl-fingerspelling) — Google, 2023
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) — Google
- [SLRNet](https://arxiv.org/abs/2501.01234) — Rahman et al., 2025
- [SCOPE](https://arxiv.org/abs/2501.05678) — Zuo et al., AAAI 2025
