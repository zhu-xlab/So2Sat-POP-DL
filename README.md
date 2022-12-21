# dl_popest_so2sat

## Model Architecture

![Baseline_Regression_Arch](https://user-images.githubusercontent.com/61827990/208144547-9f0ec1f5-4a51-4589-b0d0-4b2d692b621e.png)


## Data set
```
So2Sat POP dataset covering 98 EU cities. 
The data set has two parts. Each part can be downloaded using the following links:
So2Sat POP Part1 DOI: https://mediatum.ub.tum.de/1633792
So2Sat POP Part2 DOI: https://mediatum.ub.tum.de/1633795
Data set provides the predefined train/test split.
Randomly selected: 80% as train (80 cities) / 20% as test (18 cities)
```

## Institute
[Signal Processing in Earth Observation](https://www.asg.ed.tum.de/sipeo/home/) , Technical University of Munich, and Remote Sensing Technology Institute, German Aerospace Center.


## Funding
The work is jointly supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No. [ERC-2016-StG-714087], Acronym: \textit{So2Sat}), by the Helmholtz Association through the Framework of the Munich School for Data Science (MUDS) and the Helmholtz Excellent Professorship ``Data Science in Earth Observation - Big Data Fusion for Urban Research''(grant number: W2-W3-100), by the German Federal Ministry of Education and Research (BMBF) in the framework of the international future AI lab "AI4EO -- Artificial Intelligence for Earth Observation: Reasoning, Uncertainties, Ethics and Beyond" (grant number: 01DD20001) and by German Federal Ministry for Economic Affairs and Climate Action in the framework of the "national center of excellence ML4Earth" (grant number: 50EE2201C).

### Dependencies

Create a conda environment with python 3.8

Packages:
```
torch==1.13.1
torchvision==0.11.1
tensorboard==2.11.0
seaborn==0.12.1
matplotlib==3.6.2
opencv-python==4.6.0.66
pandas==1.1.3
scikit-learn==1.0.1
scipy==1.9.3
geopandas==0.12.1
captum==0.5.0
```


Please note that to install rasterio and GDAL, download the binary wheels for your system [rasterio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio) and [GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal). Run from the downloads folder.
```
pip install GDAL-3.4.3-cp38-cp38-win_amd64.whl
pip install rasterio-1.2.10-cp38-cp38-win_amd64.whl
```


Download the data set and add it to the current folder to run the following scripts (for both classification and regression):
```
skipt_train.py: To start the training
eval.py: to evaluat the trained model eithet on the whole data set or on individual cities.
evaluate_ghs_ours_eu.py: compute the evaluation metrics on eu cities with ghs-pop and our's predictions.
evaluate_ghs_ours_us.py: compute the metrics on ghs-pop and ours prediction on us cities.
```
