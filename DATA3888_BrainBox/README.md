# BrainBox ‑ Eye‑Movement‑Driven Page Turning

> **Course:** DATA3888 (Advanced Data Science Project)  
> **Author:** Yushang Chen  
> **Semester:** T2 ‑ 2025


## ✨ Project Overview
BrainBox is an **eye‑movement interface** that allows a user to flip digital pages **hands‑free**.  
Using a Backyard Brains **SpikerBox** to record electro‑oculography (EOG) signals, the system

1. **Collects** labelled raw voltage traces while the user looks **left**, **right**, or **blinks**;
2. **Cleans & segments** the noisy analogue signals into usable samples;
3. **Trains** a lightweight machine‑learning model that classifies the three gestures in real time;
4. **Publishes** the predictions as keyboard events so any e‑reader (e.g. PDF viewer) can turn pages automatically.

The pipeline is completely open‑source and runs on macOS / Linux / Windows with a standard Python stack.

---

## 📁 Repository Layout
```
DATA3888_BrainBox/
├── datasets/          # Raw & processed EOG recordings (CSV)
│   ├── raw/           # Untrimmed .wav & .csv files straight from SpikerBox
│   └── processed/     # 3‑second windows labelled left/right/blink
├── program/
│   ├── acquisition/   # Serial / sound‑card drivers & live preview GUI
│   ├── preprocessing/ # Filtering, baseline correction, peak detection
│   ├── training/      # Model definition, training scripts, notebooks
│   └── inference/     # Real‑time classifier + virtual‑key emitter
├── sound/             # Tone cues used during data collection
├── DEMO.mp4           # 60‑second demonstration (hands‑free page flip)
├── PROJECT3888.pdf    # Final project report
└── README.md          # You are here 🚀
```

---

## 🏃‍♀️ Quick Start
### 1. Clone & set up
```bash
# Clone
git clone https://github.com/your‑username/DATA3888_BrainBox.git
cd DATA3888_BrainBox

# Create Python environment (≥3.9 recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r program/requirements.txt
```

> **Dependencies**: `numpy`, `pandas`, `scipy`, `scikit‑learn`, `pyserial`, `sounddevice`, `matplotlib`, `pynput` (for virtual keystrokes).

### 2. Collect your own data *(optional but recommended)*
```bash
python program/acquisition/collect.py        --duration 60  \  # seconds
       --subject you           --channels 2     # mono=1, stereo=2
```
Follow the tone cues in **sound/**: look *left* on a low beep, *right* on a high beep, *blink* on a double‑beep.

### 3. Pre‑process recordings
```bash
python program/preprocessing/clean.py --input datasets/raw --output datasets/processed
```
This performs:
* 1‑45 Hz Butterworth band‑pass
* Z‑score normalisation
* Sliding‑window segmentation (3 s, 50 % overlap)
* Automatic artefact rejection (<‑15 mV / >15 mV)

### 4. Train the classifier
```bash
python program/training/train.py --data datasets/processed --model artefact/model.joblib
```
Default model: **Random Forest (100 trees)** → ~96 % accuracy on validation set.

### 5. Run live inference
```bash
python program/inference/stream.py --model artefact/model.joblib
```
* **Look left** → sends <kbd>←</kbd>   
* **Look right** → sends <kbd>→</kbd>   
* **Blink** → sends <kbd>Space</kbd>

Open any PDF/ebook reader and control it with your eyes! 👀📖

---

## 🔬 Data
| Gesture | Samples | Duration / sample | Description |
|---------|---------|-------------------|-------------|
| Left    | 1 240   | 3 s               | Eyes move from centre to far left|
| Right   | 1 230   | 3 s               | Eyes move from centre to far right|
| Blink   | 1 190   | 3 s               | Voluntary, full eyelid closure|

Raw recordings were taken at **20 kHz** through the SpikerBox audio jack, then down‑sampled to **2 kHz** post‑collection.

> *Privacy note:* All datasets contain only anonymised voltage traces; no video or personal identifiers are stored.

---

## 📈 Benchmark Results
| Model | Precision | Recall | F1‑score | Latency (ms) |
|-------|-----------|--------|----------|--------------|
| Random Forest | 0.96 | 0.95 | 0.95 | 12 ± 3 |
| SVM (RBF)     | 0.94 | 0.94 | 0.94 | 35 ± 5 |
| 1‑D CNN       | 0.98 | 0.97 | 0.97 | 8 ± 2 |

The 1‑D CNN is bundled in **artefact/cnn.h5**, but requires TensorFlow.

---

## 🛠️ Troubleshooting
| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| Flat signal | Ground not connected | Clip reference electrode to earlobe |
| High 50 Hz hum | Mains interference | Enable notch filter (`--notch 50`) |
| Key presses lag | CPU throttling | Reduce window length (`--window 1.5`) |

---

## 🎬 Demo
Watch **DEMO.mp4** for a full pipeline walk‑through and live page‑turning demo (60 s).

---

## 📄 Project Report
`PROJECT3888.pdf` contains methodology, literature review, full evaluation, and future work.

---

## 🤝 Contributing
Pull requests are welcome! Please open an issue first to discuss major changes.

1. Fork → Create feature branch → Commit → Open PR.  
2. Before submitting, run `pytest` & ensure *black*‑formatted code.

---

## 🪪 License
This project is licensed under the **MIT License**.  
See `LICENSE` for details.

---

## 📚 Acknowledgements
* Backyard Brains – SpikerBox hardware & SDK.
* UNSW Biomedical Engineering Lab for equipment & supervision.
* Open‑source libraries listed above.

---

> *“Eyes on the page, hands on nothing.”* – BrainBox
