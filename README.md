# Medical Image Segmentation
Medical image segmentation using deep learning methods


## Library
Please recovery environment by `conda`
```bash
conda env create -f pyenv.yml
```

or by `pip`
```bash
pip install -r requirements.txt
```


## Usage

Change working directory to project root directory and activate virtual environment.
```bash
cd ~/MedicalImageSegmentation
source activate <env-name>
```

### 1. Process LiTS dataset

* Convert 3D nii images and labels to 2d slices:
```bash
PYTHONPATH=./ python DataLoader/Liver/extract.py

# -------------------------------------------------------
Please choice function:
	a: exit()
	b: run_nii_3d_to_png()
	c: run_dump_hist_feature() [A/b/c] b
# Choice `b` for executing function `run_nii_3d_to_png()`
```

### 2. Train/Evaluate a UNet model

**A. For help**
```bash
python main.py --help
```

**B. Write your bash file in `./run_scripts/001_unet.sh`.**
* Required Parameters:
  * `--mode`: {train, eval}
  * `--tag`: (better to use current script name, such as ${BASE_NAME%".sh"})
  * `--model`: {UNet}
  * `--classes`: (your class names, such as "Liver Tumor")
  * `--test_fold`: which fold for validating, others for training

* Optional Parameters:
  * --zoom_scale 1.2 (for data augmentation)
  * --im_height 256 --im_width 256 (specify training image size, -1 if not specified)
  * --im_channel 1 (specify image channel)
  * --num_of_total_steps (number of max training steps)
  * --primary_metric ("<class>/<metric>", such as "Liver/Dice")
  * --weight_decay_rate (weight decay rate)
  * --learning_policy (learning rate decay policy, such as "plateau")
  * --warm_start_from "004_triplet/best_model.ckpt-215001" (absolute or relative path)

* Other Parameters:
  * Please run `python main.py --help` for details

**C. Add execution permission for your bash file: `chmod u+x ./run_scripts/train_unet.sh`**
* training
```bash
./run_scripts/001_unet.sh train 0
./run_scripts/001_unet.sh train 0,1   # Using multiple GPUs
```
* evaluation
```bash
# evaluation with best checkpoint
./run_scripts/train_unet.sh eval 0 --load_status_file checkpoint_best
# evaluation with final checkpoint
./run_scripts/train_unet.sh eval 0 --eval_final
# evaluating and saving predictions
./run_scripts/train_unet.sh eval 0 --save_predict
```
* other parameters...

### 3. How to train/eval a OSMNUNet model with your data ?

Coming soon...


## Need implemented

- [x] Data loader
- [x] Training routine
- [x] Summaries
- [x] Metrics
- [x] Loss functions
- [x] Evaluate
- [ ] Predict
- [x] Visualization
- [x] Multi-GPU Training
- [ ] Multi-GPU Evaluation/Prediction

- [x] UNet
- [ ] Attention UNet
- [ ] OSMN

- [x] Balance dataset
- [ ] Pre-trained backbone
- [ ] Spatial guide
- [x] Gaussian distribution
