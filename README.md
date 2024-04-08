# Asking the right questions for Mutagenicity prediction from Biomedical text 
#

This repository contains the source code and dataset required to generate the results available in the scientific article *Asking the right questions for Mutagenicity prediction from Biomedical text*. 


## Requirements  

The following softwares are required to run the application:

* Python 3.7 or above
* Web browser (Chrome, Firefox)

## Installation ##

After meeting the software requirements, create a dedicated virtual environment in your terminal using the command:

```
pip install virtualenv
virtualenv -p /usr/bin/python3 mutapred_bert

```

It will automatically create an environment named `mutapred_bert`. Activate and install the required packages using the following commands

```
source mutapred_bert/bin/activate
pip install -r requirements.txt

```

You can then launch Jupyter in your web browser with the command:

```
jupyter notebook

```

Open the Notebook which you want in the `notebooks` folder


## The data sets ##

We started by procuring relevant abstracts which describe chemical entities and their AMES mutagenecity. For this we used  the help of [eutils services](https://github.com/ropensci/rentrez) of PubMed. The Ames mutagenicity benchmark dataset for the chemicals and their mutagenecity comes from the publication by [Hansen et al.](https://doi.org/10.1021/ci900161g) where the column "CAS identifier" were extracted and subsequently a synonym search was performed for each molecule using the [Chemical Identifier Resolver API](https://cactus.nci.nih.gov/) provided by the NCI. Further explanation on the data procurement is mentioned in the Abstract collection section of the paper

In total there were 1,646 abstracts retrieved. Among these, there were 916 abstracts which indicated the (positive) mutagenic behaviour in their respective strains, other abstracts(730 in number) indicated the chemical which were not mutagenic. More details about the paper can be found here. The resulting dataset can be found in the `data/` directory

### About us ###

* [Sathwik Acharya](mailto:sathwik.acharya@gmail.com)
* [Sucheendra K. Palaniappan](mailto:sucheendra@sbi.jp)
* [The Systems Biology Institute](http://www.sbi.jp/)