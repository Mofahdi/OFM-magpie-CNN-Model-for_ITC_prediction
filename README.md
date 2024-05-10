# OFM-magpie-CNN-Regression
CNN model that predicts interfacial thermal conductance (ITC) using orbital field matrix (OFM) and magpie features

### to run the scripts
1. clone the repository<br>
        > `https://github.com/Mofahdi/OFM-magpie-CNN-Regression`
2. install the following packages: <br>1- sklearn, 2- seaborn, 3- matplotlib, 4- pymatgen, 5- matminer, 6- tensorflow<br>
you might have to install more packages, but you most likely have the others
3. unzip data.zip
4.  run the following in the command line: <br>
        > `python main.py --ofm_channels 32 32 64 --ofm_kernels 5 3 3 --magpie_channels 32 48 64 --magpie_kernels 3 3 3 -b 32 --lr 0.001 --epochs 50 --output_dir "results"`
    <br>Note: you have to type three integers in the *_channels and *_kernels because that is what the model requires otherwise you will get an error. 

## References
[1] "High Throughput Substrate Screening for Interfacial Thermal Management of β-Ga2O3 by Deep Convolutional Neural Network"
<br>the article should be available at https://scholar.google.com/citations?user=5tkWy4AAAAAJ&hl=en
<br>Please cite the above article
