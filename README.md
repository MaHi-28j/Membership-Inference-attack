# Membership-Inference-attack
Team Members: Mahitha Senthilnathan 7085415, Srinidhi Sreenivasan 7086703 

# Membership Inference Attack (MIA)

## Overview
This project implements a membership inference attack using a target model and a reference model.  
The goal is to determine whether a data sample was part of the training set of the target model.


## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- tqdm

Install dependencies:

pip install torch torchvision numpy pandas tqdm


## Project Structure

Place the following files in the same directory:

- pub.pt          (public dataset with membership labels)
- priv.pt         (private dataset for submission)
- model.pt        (trained target model)
- script.py       (this script)


## How to Run

1. Open a terminal in the project directory

2. Run the script:

python script.py


## What the Script Does

1. Loads the public and private datasets
2. Splits the public dataset into members and non-members
3. Trains a reference model using non-member data
4. Loads the target model
5. Computes membership scores for both datasets
6. Evaluates performance on the public dataset
7. Generates a submission file


## Output

- submission.csv

This file contains:

- id: sample identifier
- score: normalized membership score


## Notes

- Set RETRAIN_REF = False to reuse a saved reference model
- GPU will be used automatically if available
