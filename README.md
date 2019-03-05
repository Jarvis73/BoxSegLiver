# Medical Image Segmentation
Medical image segmentation using deep learning methods


## Library
```bash
conda create -n xxx python=3.6 tensorflow-gpu=1.12 scipy matplotlib pathlib traits traitsui scikit-image
conda install -c conda-forge pyyaml nibabel
pip install medpy
```


## Usage

### 1. How to construct a tf-record dataset for your own data ?

* Create a subfolder in `./data/` for your dataset. For example `./data/new_dataset/`.
* Put your dataset folder into `new_dataset`. For example `./data/new_dataset/TrainingImages/`.
* Create a text file saving image list. For example `./data/new_dataset/trainval.txt`.
  * All the image paths in `trainval.txt` should be based on `./data/new_dataset/`:
```
TrainingImages/image-1.png
TrainingImages/image-2.png
TrainingImages/image-3.png
......
```
* Write your own `build_xxx.py` following the format of `./data_kits/build_lits_liver.py`.
* Add your script file `./data_kits/build_xxx.py` to `./build_datasets.py` and run `python build_datasets.py`
* Write a json file like `./data/LiTS_Train.json` if you have multiple tf-records for training/evaluation.

### 2. How to train/eval a UNet model with your data ?

**A. For help**
```bash
python main.py --help
```

**B. Write your bash file `train_unet.sh` in `./run_scripts/`.**
* Required Parameters:
  * --mode {train, eval}
  * --tag (use script name)
  * --model {UNet}
  * --classes (your class name, a list)
  * --dataset_for_train
  * --dataset_for_eval

**C. Add execution permission for your bash file: `chmod u+x ./run_scripts/train_unet.sh`**
* training
```bash
./run_scripts/train_unet.sh train 0
```
* evaluation
```bash
./run_scripts/train_unet.sh eval 0
```
* evaluation with saving predictions
```bash
./run_scripts/train_unet.sh eval 0 --save_predict
```
* other parameters...


## Need implemented

- [x] Data loader
- [x] Training routine
- [x] Summaries
- [x] Metrics
- [x] Loss functions
- [x] Evaluate
- [x] Predict
- [ ] Visualization

- [ ] UNet
- [ ] ResUnet
- [ ] Atrous Conv Nets

- [ ] Pre-trained backbone
- [x] Spatial guide network
- [ ] Classifier for Gaussian distribution


## Notice

* Modify Xception: First conv stride = 1

