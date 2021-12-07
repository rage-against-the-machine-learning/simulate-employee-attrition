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
├── arena
│   ├── ARENA-output.pdf
│   └── Attrition-Model-v-2021-30-11.doe
├── data             # hidden file directory, download data from kaggle
│   └── raw
│       └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── environment.yaml # file to create virtual conda environment
├── notebooks
│   ├── IBM-Data-EDA-for-SIM.ipynb   # statistical analysis, EDA, plots
│   └── LogReg.ipynb # work & results for logistic regression
├── reports
│   ├── figures      # plots generated
│   └── README.md    # data dictionary
└── src
    ├── config.py    # global variables 
    ├── dectree.py   # decision tree code
    ├── helper.py    # data preprocessing & prep 
    └── logreg.py    # logisitc regression binary classifier
```

### References:
Data Source: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
