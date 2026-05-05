# Membership Attack Inference 
Team Members: Mahitha Senthilnathan 7085415, Srinidhi Sreenivasan 7086703

## 1. Environment Setup

Install required dependencies:
```
pip install torch torchvision numpy pandas tqdm
```

Use Python 3.8+.
GPU is recommended for faster execution.


## 2. Project Files

Place the following files in the same directory as the script:

- pub.pt
- priv.pt
- model.pt
- script.py

The script automatically uses the current directory.  
If needed, modify the BASE path at the top of the script.


## 3. Configuration

Ensure the following settings in the script:
```
RETRAIN_REF = True
```

This trains the reference model from scratch, which is required to reproduce the same results.

After the first run, a file will be saved:

- ref_model.pt

For faster future runs, you may set:

```
RETRAIN_REF = False
```

This will load the saved reference model instead of retraining.


## 4. Hardware

- GPU (CUDA) will be used automatically if available
- CPU can be used but will be significantly slower


## 5. Run the Script

Execute:

python script.py
