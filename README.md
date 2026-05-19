# NGTFlow-Neuroplasticity-Guided-TimeFlow-for-Longitudinal-Depression-Recovery
Neuroplasticity Guided TimeFlow(NGTFlow): A Longitudinal MRI Approach to Depression Recovery

NGTFlow/
├── README.md
├── requirements.txt
├── setup.py
├── main.py
├── ngtflow/
│   ├── __init__.py
│   ├── model.py
│   ├── ode.py
│   ├── loss.py
│   ├── hippocampus.py
│   ├── data.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── infer.py
│   └── evaluate.py
├── configs/
│   └── default.yaml
├── notebooks/
│   └── visualization.ipynb
├── results/
│   └── figures/
└── .gitignore

# NGTFlow: Neuroplasticity Guided TimeFlow

**The first fully continuous, time-conditioned neural flow model for longitudinal neuroplasticity in depression recovery.**

Official implementation of the paper:  
**"Neuroplasticity Guided TimeFlow (NGTFlow): A Longitudinal MRI Approach to Depression Recovery"**

## Key Features
- Continuous-time Neural ODE-based diffeomorphic deformation
- Biologically motivated hippocampal volume & shape constraint
- Joint anatomical reconstruction + symptom (MADRS) trajectory prediction
- Handles irregular scan intervals and intervention schedules

## Installation

```bash
git clone https://github.com/AbhishekTiwari101/NGTFlow.git
cd NGTFlow
pip install -e .

# Train
python scripts/train.py --config configs/default.yaml

# Inference
python scripts/infer.py --baseline_mri path/to/baseline.nii.gz --time_months 12

