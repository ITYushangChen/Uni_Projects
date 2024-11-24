# Trends across the week in step activity

Group: 2022-DATA3406-CC03-Group03

Team members: Emily Cai, Yushang Chen, Ziyang Lin, Callum Smith, Amelia Xie

## Driving Question

This project is an analysis on the individual step activity patterns, with a focus on the driving question of what are the trends across the week. We will be taking a deeper look at possible influences on step activity and evaluating their impact on the trends in step counts across the week.


![image](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/images/2010.i506.007.city%20composition%20flat.jpg)

## Notebook requirements

To ensure the reproducibility of our notebook, we have provided a list of packages used and their respective versions (**NOTE:** the package versions listed below were tested to run on the Product Notebook. Using package versions other than those specified in this list may result in undefined behaviour(s)):

- `numpy 1.23.3`
- `pandas 1.4.4`
- `matplotlib 3.5.2`
- `seaborn 0.12.0`
- `plotly 5.9.0`
- `pathlib 1.0.1`
- `ipywidgets 7.6.5`
- `scipy 1.9.1`
- `itertools (built-in module)`

## Table of Contents

* [Driving Question](#Driving-Question)
* [Notebook requirements](#Notebook-requirements)
* [Key Definition](#Key-Definition)
* [Individual Process Notebooks](Process-Notebooks)
* [Product Notebook](Product-Notebooks)
* [Ethical Analysis](#Ethical-Analysis)
* [Use of Issues and Wiki](#Use-of-Issues-and-Wiki-for-Group-Processes)
* [Reflection on group processes](#Reflection-on-Group-Processes)
* [Think Aloud and Cognitive Walkthrough](#Think-aloud-and-cognitive-walkthroughs)
* [Testing for Product Notebook](#Testing-for-Product-Notebook)


Key Definition
---------
**Key Variables**

Most of the datasets provided by Professor Judy Kay are of the structure below:
|Start|Finish|Steps (count)|
|:---|:---|:---|
|Starting date and time|Ending date and time|Step counts recorded|

This is the structure for the detailed datasets:
|Source|Date|Hour|Count |
|:---|:---|:---|:---|
|Source of data (For participant 2, this is the device used to track step counts. For participant 5, this is the ID of the participant)|Date data was collected|Hour|Step counts recorded|



For information regarding the variables of the additional dataset, here is the metadata:

[Rainfall metadata](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/datasets/additional/rainfall_syd_meta.txt)

[Solar exposure Metadata](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/datasets/additional/rainfall_cp_meta.txt)

**Folder details**

1. [datasets folder](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/tree/main/datasets)

This folder contains all the datasets that was used in this analysis, along with datasets that was not used but was considered for this project

2. [additional dataset](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/tree/main/datasets/additional)

This folder contains datasets that was not provided at the start of this project

3. [images folder](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/tree/main/images)

This folder contains all the images that was used in the README and our Wiki

4. [notebooks folder](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/tree/main/notebooks)

This folder contains all the process and product notebooks produced from this project

5. [Process notebook folder](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/tree/main/notebooks/Process%20Notebooks)

This folder contains all our process notebooks

6. [Product notebook folder](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/tree/main/notebooks/Product%20Notebooks)

This folder contains all our result notebooks

Process Notebooks
------------------------


Amelia
- [Initial Data Exploration P4](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Week08-Amelia_Initial_data_exploration_dataset04.ipynb)
- [Initial Data Exploration P7](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Week10-Amelia_Initial_data_exploration_dataset07.ipynb)
- [Inital Data Exploration P8](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Week10-Amelia_Initial_data_exploration_dataset08.ipynb)
- [Sub Question Analysis P4](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Amelia_sub_question_analysis_(participant_4).ipynb)
- [Sub Question Analysis P7](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Amelia_sub_question_analysis_(participant_7).ipynb)
- [Sub Question Analysis P8](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Amelia_sub_question_analysis_(participant_8).ipynb)
- [Process Notebook Week 10](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Amelia_TPP10_notebook.ipynb)
- [Process Notebook Week 11](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Amelia_Process_Notebook_Week%2011.ipynb)
- [Process Notebook : Personal Report ](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Amelia%20Xie/Personal%20report%20p4%20code%20replication.ipynb)
- [Process Notebook : Work for Product Notebook ](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Product%20Notebooks/Product_Notebook_draft_Amelia's%20version.ipynb)

Emily 
- [Initial Data Exploration P1](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Emily%20Cai/Week08-Emily-Initial-Analysis.ipynb)
- [Sub Question Exploration P1](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Emily%20Cai/Week09_Emily.ipynb)
- [Initial Exploration and Sub Question Exploration: P1](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Emily%20Cai/TPP10_P1_Emily.ipynb)
- [Initial Exploration and Sub Question Exploration: P2](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Emily%20Cai/TPP10_Emily.ipynb)
- [Initial Exploration and Sub Question Exploration: P5](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Emily%20Cai/TPP10_P5_Emily.ipynb)
- [Sub Question that was dropped](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Emily%20Cai/Corr_P125_Emily.ipynb)
- [Process Notebook Week 10](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Emily%20Cai/Emily_Progress_Notebook.ipynb)
- [Process Notebook Week 11](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Emily%20Cai/Draft_Product_Notebook_Emily.ipynb)
- [Process Notebook Week 12](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Product%20Notebooks/Draft_Product_Notebook_Emilyv2%20-%20Copy.ipynb)

YuShang
- [Exploration P7](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Yushang%20Chen/Exploration.ipynb)
- [Process Notebook P7](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Yushang%20Chen/Paticipants7_Analysis.ipynb)
- [Process Notebook P3](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Yushang%20Chen/Paticipants3_Analysis.ipynb)
- [Process Notebook Week 10](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Yushang%20Chen/week10-Yushang.ipynb)
- [Individual Report](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Yushang%20Chen/Individual_Report.ipynb)
- [Reproducible code](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Yushang%20Chen/Reproducible_Code.ipynb)
- [Process Notebook Week 12](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Yushang%20Chen/Yushang_modified.ipynb)
- [Testing for Product Notebok](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Product%20Notebooks/Testing.ipynb)
- [Additional Dataset - Dropped](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Yushang%20Chen/AdditionalDataset.ipynb)

Ziyang
- [Data Exploration](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Ziyang%20Lin/Week08-Lin%20-%20Data%20exploration%20-%20dataset02.ipynb)
- [Process Notebook Week 10](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Ziyang%20Lin/Week10-Lin.ipynb)
- [Process Notebook Week 11](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Ziyang%20Lin/Week11-Lin-additional-datasets.ipynb)
- [Process Notebook Week 12](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Ziyang%20Lin/subquestion-for-product-notebook.ipynb)

Callum
- [Data Exploration](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Callum%20Smith/Week8-Callum-Data-Exploration.ipynb)
- [Process Notebook Week 10](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Callum%20Smith/TPP10-Callum-Data-Analysis.ipynb)
- [Process Notebook Week 11 & 12](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Process%20Notebooks/Callum%20Smith/TPP11-Callum-Participant-03.ipynb)

Product Notebooks
---------

- [Draft Product Notebook](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Product%20Notebooks/Draft_Product_Notebook_Emily.ipynb)
- [Product Notebook](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/blob/main/notebooks/Product%20Notebooks/Main_Product_Notebook.ipynb)



Ethical Analysis
-------------


[Ethical Analysis Wiki](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Ethical-Considerations)


Use of Issues and Wiki for Group Processes
-----------------------

Here is the link to Our Github Wiki Minutes that documented our group meetings:

[Minutes](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Meeting-Minutes)

Our Individual Work Overviews (Contains links to relevant issues and notebook codes that were completed individually)

Amelia [Individual Summary](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Amelia's-Individual-Work-Summary)

Callum [Individual Summary](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Individual-Group-Summary---Callum)

Emily [Individual Summary](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Individual-Summary-Emily-Cai)

Yushang [Individual Summary](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Individual-Work-Yushang)

Ziyang [Individual Summary](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Individual-work-summary----Ziyang-Lin)

Our Individual Group Roles and Group Contract:

[Group Contract](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Group-Contract)

[Group Roles](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Group-Roles)


Reflection on Group processes
------------------

[Reflection](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Reflection-on-group-processes)


Think Aloud and Cognitive Walkthroughs
------------------

[Summary of Think Alouds](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Summary-of-think-alouds)

[Summary of Cognitive Walkthrough](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Summary-of-cognitive-walkthrough)

[Reflection on Think Aloud and Cognitive Walkthrough](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Reflection-on-lessons-learnt-from-think-aloud-and-cognitive-walkthrough)

Testing for Product Notebook
------------------

[Testing Wiki Page](https://github.sydney.edu.au/zlin4387/2022-DATA3406-CC03-Group03/wiki/Testing)

