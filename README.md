# **Online supplement**: A cultural evolutionary model for the law of abbreviation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11355636.svg)](https://doi.org/10.5281/zenodo.11355636)

Authors: 

- Olivier Morin
- Alexey Koshevoy



## Reproduction 

### Downloading the code & requirements:

The code provided in this repository was executed using Python 3.11.7. First, clone the repository:

```bash
git clone https://github.com/alexeykosh/2023-dripf-model/tree/main
```

Then, navigate to the repository:

```bash
cd 2023-dripf-model
```

All the required packages are listed in the `requirements.txt` file. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Data:

Data for the empirical results can be downloaded from the following [link](http://www.christianbentz.de/AdaptiveLanguages/AdaptLang.zip). This data was used in the [Bentz & Ferrer-i-Cancho, 2016](https://publikationen.uni-tuebingen.de/xmlui/handle/10900/68639) paper. The data is stored in the `FreqDists_50K` folder, which needs to be put in the `data` folder of this repository.

### Code:

- [empirical data](https://github.com/alexeykosh/2023-dripf-model/blob/main/notebooks/empirical_data.ipynb) contains the code for the reproduction of Figures 1 & 2 from the paper that are based on the empirical data.

- [model](https://github.com/alexeykosh/2023-dripf-model/blob/main/notebooks/model.ipynb) contains the code for the reproduction of Figures 3 & 4 from the paper that are based on the cultural evolutionary model of the law of abbreviation. The code for the model is implemented in the same notebook.
