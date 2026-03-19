# OmniSub2026 — Visual Speech Recognition

> Kaggle Competition: **omni-sub**
> Task: Transcribe silent lip-reading videos into English text
> Metric: Word Error Rate (WER) — lower is better

---

## What This Does

This notebook uses **AutoAVSR** — a state-of-the-art Visual Speech Recognition model from Imperial College London — pretrained on the LRS3 dataset (19.1% WER). It:

1. Downloads the competition data directly from Kaggle
2. Downloads the pretrained model and language model automatically
3. Runs inference on all 49 test videos
4. Submits predictions directly to Kaggle

No GPU on your local machine? No problem — runs entirely on **Google Colab's free T4 GPU**.

---

## Quickstart (for teammates)

### Step 1 — Get your Kaggle API token
1. Go to [kaggle.com](https://www.kaggle.com) → click your profile → **Settings**
2. Scroll to **API** → click **Create New Token**
3. Copy the token (looks like `KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

### Step 2 — Open the notebook in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/OmniSub2026/blob/main/OmniSub2026_Colab.ipynb)

Or manually: go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** → paste this repo URL.

### Step 3 — Set runtime to T4 GPU
**Runtime → Change runtime type → T4 GPU**

### Step 4 — Run the notebook
1. Paste your Kaggle API token in **Step 1** of the notebook
2. Click **Runtime → Run all**
3. Wait ~30 minutes for everything to download and run
4. Submission is automatically sent to Kaggle at the end

---

## What Downloads Automatically

| File | Size | Source |
|------|------|--------|
| Competition data (train + test) | ~varies | Kaggle API |
| VSR Model weights (LRS3 WER 19.1%) | ~955 MB | Google Drive |
| Language model (English subword) | ~191 MB | Google Drive |
| AutoAVSR source code | ~few MB | GitHub |

---

## Project Structure

```
OmniSub2026/
├── OmniSub2026_Colab.ipynb          # Main notebook — run this on Colab
├── OmniSub2026_VSR_Guide-1.md       # Full competition guide
├── run_inference.py                 # Local CPU inference script (no GPU needed)
├── README.md                        # This file
├── data/
│   ├── train/<video_id>/<clip>.mp4  # Training videos
│   ├── train/<video_id>/<clip>.txt  # Ground truth transcripts
│   ├── test/<clip>.mp4              # Test videos to predict
│   └── sample_submission.csv        # Submission format
├── checkpoints/pretrained/          # Local model weights
└── Visual_Speech_Recognition.../   # AutoAVSR repo (cloned locally)
```

---

## Running Locally (No GPU / CPU Only)

If you want to run inference on your own machine:

```bash
# 1. Activate environment
source vsr_env/bin/activate

# 2. Run batch inference on all 49 test videos
python run_inference.py

# 3. Submit result
KAGGLE_API_TOKEN=your_token kaggle competitions submit \
    -c omni-sub -f data/submission.csv -m "local CPU inference"
```

> **Note:** CPU inference takes ~4 minutes per video (~3-4 hours total for 49 videos).

---

## Model Details

| Property | Value |
|----------|-------|
| Model | AutoAVSR Conformer |
| Pretrained on | LRS3 + VoxCeleb2 |
| Modality | Video only (VSR) |
| WER on LRS3 test | 19.1% |
| Detector | MediaPipe (face/lip) |
| Decoder | Beam search (size=40) + RNN-LM |
| Authors | Imperial College London |
| Paper | [Visual Speech Recognition for Multiple Languages](https://arxiv.org/abs/2202.13084) |

---

## Improving the Score (Fine-tuning)

After getting a baseline submission, you can fine-tune on the competition's training data:

1. Uncomment **Step 9** in the notebook
2. This fine-tunes the pretrained model on the 1603 training clips
3. Re-run inference and submit again

Fine-tuning for even 3-5 epochs on a T4 GPU typically reduces WER by 5-15%.

---

## Team

- Competition: [omni-sub on Kaggle](https://www.kaggle.com/competitions/omni-sub)
- Model credit: [mpc001/Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
