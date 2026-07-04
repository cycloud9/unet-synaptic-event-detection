# U-Net Synaptic Event Detection

A deep-learning-based graphical tool for detecting synaptic events in electrophysiology recordings.  
This project provides U-Net inference pipelines and a Tkinter GUI for EPSC and IPSC event detection, review, correction, and export.

The software is designed for patch-clamp current traces and supports both automatic event detection and manual inspection through an interactive trace viewer.

---

## Overview

This repository contains the source code for a U-Net-based synaptic event detection system.

Main features:

- EPSC and IPSC detection modes
- U-Net probability-based event detection
- Support for CSV trace input
- Support for ABF-to-CSV conversion
- Interactive GUI for event visualization
- Event list navigation
- Potential event review
- Manual event promotion and demotion
- Single-event zoom view
- Export of confirmed events to CSV

The GUI is intended to help users combine automated deep learning inference with manual electrophysiology event curation.

---

## Repository Structure

```text
unet-synaptic-event-detection/
├── app/
│   ├── unet_gui_v8.py
│   ├── unet_ola_fp_fn.py
│   ├── unet_ola_fp_fn_ipsc.py
│   ├── abf2csv.py
│   └── __init__.py
├── docs/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── assets/
├── requirements.txt
├── run_gui.py
└── README.md
