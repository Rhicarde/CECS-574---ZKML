# Medical Privacy Preservation using ZKML

## Overview
This project demonstrates the concept of Zero Knowledge Proofs and how it can be applied to machine learning models. In this project, we use EZKL in order to convert our logistic regression model into a ZKP circuit which can be used to create proofs on given input. The dataset used to trained the model is for Heart Disease Risk Prediction.
## Features
- Privacy-Perserving Machine Language Model: Ensure data remains private and localized, complying with health regulations
- Witness and Verifier Generation: Creates proof and model output to be verified either with the WebAssembler or on-chain
  
## Project Architecture
![image](https://github.com/user-attachments/assets/bf998093-8760-467a-8a13-7d5f6f446152)

## Instructions
To run the model, download the .ipynb file and run it on a notebook of your choice. Using Google Colab does make the process easier as you will not have to manually install the dependencies.

## System Requirements
Project is done in a notebook so ensure that required libraries are installed.
### Dependencies
- Python 3.10
- Required Libaries:
  - `google.colab`
  - `kagglehub`
  - `pandas`
  - `sklearn-kit`
  - `torch`
  - `ezkl`
  - `onnx`
  - `networkx`
  - `subprocess`
  - `sys`
  - `os`
  - `json`
  - `matplotlib`
