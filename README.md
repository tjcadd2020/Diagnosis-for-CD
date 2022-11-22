![](https://img.shields.io/badge/Python-3.8-brightgreen) ![](https://img.shields.io/badge/R-4.1.2-lightgrey) ![](https://img.shields.io/badge/Version-dev-yellowgreen) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/github-gs/Diagnosis-for-CD/HEAD) ![](https://img.shields.io/badge/lisense-MIT-orange) [![](https://img.shields.io/badge/Website-CADD-blue)](https://cadd.tongji.edu.cn/)

# Diagnosis-for-CD
## *Microbial genes outperform species and SNVs as diagnostic markers for Crohn’s disease according to artificial intelligence analyses of multicohort fecal metagenomes*  

Our global metagenomic analysis unravels the multi-dimensional alterations of the microbial communities in CD, and identified microbial genes as robust diagnostic biomarkers across cohorts. These genes are functionally related to CD pathology. Future research on these genes may lead to an effective non-invasive diagnostic tool for CD.  
  
  

    
### Descriptions for scripts

### 1.Raw data process  

Metagenomic sequencing data:  

	(1) Quality control  

	(2) Taxonomic annotation and abundance estimation  

	(3) Gene prediction and abundance estimation  

### 2.SNV Calling  

MIDAS was used to perform microbial SNV annotation.  
The WMS reads were mapped to the database for SNV calling.  

### 3.Differential analysis with MMUPHin  

Identifying differential signatures from multi-cohorts by correcting batch effects.  

### 4.Alpha diversity  

Calculating alpha diversity of microbiome in CD patients and controls.

### 5.Beta diversity  

Calculating beta diversity of microbiome in CD patients and controls.

### 6.Model contruction with FNN  

Feedforward neural network (FNN) was employed to construct the diagnostic model:  
	
	(1) Neuron network construction  

	(2) Ten fold cross-validation  

	(3) Model trainning   

	(4) Model testing  

### 7.Feature evaluation with SHAP  

The feature importance was evaluated with SHapley Additive exPlanations (SHAP) to explain the output of machine learning model.