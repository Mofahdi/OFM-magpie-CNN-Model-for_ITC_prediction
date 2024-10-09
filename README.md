# OFM-magpie-CNN-Model-for_ITC_prediction
CNN model with fused orbital field matrix (OFM) and magpie features that predicts interfacial thermal conductance (ITC).

### to run the scripts
1. clone the repository<br>
        > `git clone https://github.com/Mofahdi/OFM-magpie-CNN-Model-for_ITC_prediction.git`
2. install the following packages: 
<br>1- sklearn 
<br>2- seaborn 
<br>3- matplotlib 
<br>4- pymatgen 
<br>5- matminer 
<br>6- tensorflow<br>
you might have to install other packages, but you most likely have them.
3. unzip data.zip
4.  run the following in the command line: <br>
        > `python main.py --ofm_channels 32 32 64 --ofm_kernels 5 3 3 --magpie_channels 32 48 64 --magpie_kernels 3 3 3 -b 32 --lr 0.001 --epochs 50 --output_dir "results"`
    <br>Note: you have to type three integers in the *_channels and *_kernels because that is what the model requires otherwise you will get an error. 

## Credit & References
[1] Al-Fahdi, M.; Hu, M. High Throughput Substrate Screening for Interfacial Thermal Management of β-Ga2o3 by Deep Convolutional Neural Network. **Journal of Applied Physics** *2024*, 135 (20).
<br>

* the code was used to produce the work in the above article which should be available at this [link](https://digital.csic.es/bitstream/10261/360878/1/AlFahdi_ApplPhysRev_2024_editorial.pdf). Please cite the above article if you use the code
* Please consider reading my published work in Google Scholar using this [link](https://scholar.google.com/citations?user=5tkWy4AAAAAJ&hl=en&oi=ao) thank you :)
