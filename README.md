# rossmann-pharma-sales-prediction

A machine learning solution to forecast sales for Rossmann Pharmaceuticals' stores across various cities six weeks in advance. Factors like promotions, competition, holidays, seasonality, and locality are considered for accurate predictions.
The project structure is organized to support reproducible and scalable data processing, modeling, and visualization.

## **PROJECT FOLDER STRUCUTR**
```
|   .gitignore
|   projectStructure.txt
|   README.md
|   requirements.txt
|   
+---.github
|   \---workflows
+---.venv-w4
|          
+---.vscode
|       settings.json
|       
+---Data
+---notebooks
|       __init__.py
|       
+---scripts
|       __init__.py
|       
+---src
|       __init__.py
|       
\---tests
        __init__.py
        
```
# Installation

>>> git clone https://github.com/Jenber-Ligab/Pharmaceuticals

>> cd Pharmaceuticals

### Create virtual environment

>>> python3 -m venv .venv-w4 # on MacOs or Linux

>>> source .venv-w4/bin/activate  # On Windows: venv\Scripts\activate

### Install Dependencies

>>> pip install -r requirements.txt

## To run tests
navigate 
>>> cd tests/

>>pytest # all tests will be tested
