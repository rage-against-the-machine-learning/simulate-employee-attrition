# Simulating Employee Attrition

### Using this Repository
Requirements; `Python 3.9`
1. Have [conda or miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed
2. `git clone` or download the repository
3. In the project directory in terminal: 
```markdown
$ conda env create -f environment.yaml
$ conda activate sim-proj
$ pip install simpy
```
4. Optional: Setup use of Jupyter Notebook
   (pass these commands in order, in terminal)
```markdown
$ pip install --user ipykernel
$ python -m ipykernel install --user --name=sim-proj

# check that the kernel is installed, 
# sim-proj should be listed after this command
$ jupyter kernelspec list
```

### Repository Structure:
```sh
.
├── README.md
├── data             # hidden file directory, download data from kaggle
│   ├── interim
│   ├── processed
│   └── raw
│       └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── environment.yaml
├── notebooks
│   └── 2021-10-16-Data-EDA.ipynb
└── src
    └── helper.py
```

### References: 
Data Source: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
