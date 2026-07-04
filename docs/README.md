# U-Net Synaptic Event Detection

A deep-learning-based graphical tool for detecting synaptic events in electrophysiology recordings.

This repository provides U-Net inference pipelines and a Tkinter-based graphical user interface for EPSC and IPSC event detection, visualization, manual review, correction, and export.

The software is designed for patch-clamp current traces and aims to combine automated deep learning inference with human-in-the-loop electrophysiology event curation.

---

## Project Overview

Synaptic event detection is a key step in electrophysiology data analysis. Traditional event detection methods often require manual threshold tuning and may be sensitive to baseline drift, noise, event overlap, and waveform variability.

This project implements a U-Net-based event detection workflow for synaptic current recordings. The model outputs event probability traces, which are then converted into detected events through post-processing. The GUI allows users to inspect detected events, review potential missed events, manually promote or reject detections, and export curated results.

---

## Main Features

- EPSC and IPSC detection modes
- U-Net probability-based event detection
- Support for CSV trace input
- Support for ABF-to-CSV conversion
- Interactive trace visualization
- Event list navigation
- Potential event display
- Manual event promotion and demotion
- Single-event zoom view
- Export of confirmed events to CSV
- Windows executable release for users without Python setup

---

## Repository Structure

```text
unet-synaptic-event-detection/
├── app/
│   ├── __init__.py
│   ├── abf2csv.py
│   ├── unet_gui_v8.py
│   ├── unet_ola_fp_fn.py
│   └── unet_ola_fp_fn_ipsc.py
├── docs/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── assets/
│       ├── architecture.png
│       ├── benchmark.png
│       ├── dilated.png
│       ├── gui.png
│       └── manuscript.pdf
├── requirements.txt
├── run_gui.py
├── .gitignore
└── README.md
```

---

## Project Webpage

The project webpage is available at:

```text
https://cycloud9.github.io/unet-synaptic-event-detection/
```

The webpage provides a visual summary of the model architecture, GUI, benchmark results, and project background.

---

## Windows Executable

A prebuilt Windows executable is available from the GitHub Releases page:

```text
https://github.com/cycloud9/unet-synaptic-event-detection/releases
```

The Windows release is recommended for users who do not want to install Python, TensorFlow, or other dependencies manually.

Typical release package:

```text
UNet_Synaptic_Event_Detector_v1.0.0_Windows_x64/
├── UNet_Synaptic_Event_Detector.exe
├── best_ens2_clean.keras
├── best_ipsc_slowkinetics.keras
└── README.txt
```

To run the executable:

1. Download and unzip the release package.
2. Double-click `UNet_Synaptic_Event_Detector.exe`.
3. Select EPSC or IPSC mode.
4. Click `Select Model File`.
5. For EPSC detection, select `best_ens2_clean.keras`.
6. For IPSC detection, select `best_ipsc_slowkinetics.keras`.
7. Select a trace CSV or ABF file.
8. Click `Run U-Net Inference`.
9. Review detected events and potential events.
10. Export confirmed events to CSV.

The application may take several seconds to start because TensorFlow and the GUI components need to load.

---

## Installation from Source

Users who want to run or modify the source code can install the project manually.

### 1. Clone the repository

```bash
git clone https://github.com/cycloud9/unet-synaptic-event-detection.git
cd unet-synaptic-event-detection
```

### 2. Create a Python environment

A Python 3.10 environment is recommended.

```bash
conda create -n unet-events python=3.10
conda activate unet-events
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If ABF support is needed and `pyabf` is not already installed, run:

```bash
pip install pyabf
```

---

## Requirements

Main dependencies include:

```text
tensorflow==2.15.0
keras==2.15.0
numpy==1.26.4
scipy==1.15.2
h5py==3.13.0
pandas==2.2.3
matplotlib==3.8.4
tqdm==4.67.1
protobuf==4.25.8
tensorboard==2.15.2
tensorflow-estimator==2.15.0
pyabf==2.3.8
```

If your local `requirements.txt` does not include `pyabf`, add:

```text
pyabf==2.3.8
```

This is required for loading and converting `.abf` electrophysiology files.

---

## Model Files

The trained model files are not stored directly in the source repository because they are large binary files.

Expected model files:

```text
best_ens2_clean.keras
best_ipsc_slowkinetics.keras
```

Recommended local folder structure:

```text
unet-synaptic-event-detection/
├── models/
│   ├── best_ens2_clean.keras
│   └── best_ipsc_slowkinetics.keras
```

The GUI also allows users to select model files manually, so the model files do not have to be stored in a fixed location.

---

## Running the GUI from Source

From the repository root, run:

```bash
python run_gui.py
```

Alternatively, run the GUI file directly:

```bash
python app/unet_gui_v8.py
```

If the GUI starts correctly, users can select the detection polarity, load the corresponding model file, select a trace file, and run U-Net inference.

---

## Input Format

The recommended input format is a CSV file with time and current columns.

Example:

```csv
time_s,current_pA
0.0000,-2.31
0.0001,-2.45
0.0002,-2.50
0.0003,-2.37
```

Recommended format:

- First column: time in seconds
- Second column: current amplitude in pA

The GUI also supports ABF files through the included ABF conversion utility when `pyabf` is installed.

---

## Output

The GUI exports curated event results to CSV.

The exported file may include information such as:

- Event time
- Event amplitude
- Event type
- Detection status
- User-confirmed event status
- Potential event status

The exact output columns depend on the GUI export function.

---

## EPSC and IPSC Modes

The GUI includes a polarity selector for EPSC and IPSC detection.

### EPSC Mode

EPSC mode uses the EPSC-trained U-Net model:

```text
best_ens2_clean.keras
```

This mode is intended for inward synaptic current events, typically shown as negative-going peaks depending on recording convention.

### IPSC Mode

IPSC mode uses the IPSC-trained U-Net model:

```text
best_ipsc_slowkinetics.keras
```

This mode is intended for IPSC event detection and uses the corresponding IPSC inference pipeline.

---

## GUI Workflow

A typical workflow is:

1. Open the GUI.
2. Select EPSC or IPSC mode.
3. Select the corresponding U-Net model file.
4. Load a CSV or ABF trace file.
5. Run U-Net inference.
6. Inspect detected events on the trace.
7. Switch between detected events, potential events, or all events.
8. Use event navigation controls to inspect individual events.
9. Promote potential events if they should be counted.
10. Demote false positives if they should be removed.
11. Export the curated event list to CSV.

---

## Notes on Large Files

The source repository intentionally excludes large binary and data files, including:

```text
*.exe
*.zip
*.keras
*.abf
raw trace files
exported event CSV files
```

These files should not be committed directly to the source repository.

Recommended distribution method:

- Source code: GitHub repository
- Project webpage: GitHub Pages
- Executable and trained model files: GitHub Releases
- Raw electrophysiology data: external storage or institutional data repository

---

## Development Notes

This project contains separate inference scripts for EPSC and IPSC detection:

```text
app/unet_ola_fp_fn.py
app/unet_ola_fp_fn_ipsc.py
```

The GUI entry point is:

```text
app/unet_gui_v8.py
```

The top-level launcher is:

```text
run_gui.py
```

The ABF conversion utility is:

```text
app/abf2csv.py
```

---

## Known Limitations

- Model files must be provided separately.
- The Windows executable may take several seconds to launch.
- Very large trace files may require additional loading time.
- GPU acceleration depends on the user’s TensorFlow installation and hardware environment.
- Event detection quality depends on recording quality, noise level, sampling rate, and similarity between user data and the model training distribution.

---

## Citation

If you use this software or build upon this project, please cite the associated manuscript, project webpage, or repository when available.

Suggested repository citation:

```text
U-Net Synaptic Event Detection. GitHub repository:
https://github.com/cycloud9/unet-synaptic-event-detection
```

---

## License

Please see the repository license file for usage terms.

If no license file is provided, all rights are reserved by default until a license is added.

---

## Contact

For questions, bug reports, or feature requests, please open an issue on GitHub:

```text
https://github.com/cycloud9/unet-synaptic-event-detection/issues
```
