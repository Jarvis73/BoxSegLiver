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
	c: run_dump_hist_feature()
	d: run_simulate_user_prior [A/b/c/d] b
# Choice `b` for executing function `run_nii_3d_to_png()`
```

### 2. Train/Evaluate a UNet model

**A. For help**
```bash
python ./entry/main.py --help
```

**B. Write your bash file in `./run_scripts/my_config.sh`.**

You can copy and modify provided template bash file in `./run_scripts/template/001_unet.sh`.

* Required Parameters:
  * `--mode`: {train, eval}
  * `--tag`: (better to use current script name, such as ${BASE_NAME%".sh"})
  * `--model`: {UNet}
  * `--classes`: (your class names, such as "Liver Tumor")
  * `--test_fold`: which fold for validating, others for training

* Optional Parameters:
  * `--random_flip`: 3 --noise_scale 0.05 (data augmentation)
  * `--im_height`: 256 --im_width 256 (specify training image size, -1 if not specified)
  * `--im_channel`: 3 (specify image channel)
  * `--num_of_total_steps`: (number of max training steps)
  * `--primary_metric`: ("<class>/<metric>", such as "Liver/Dice")
  * `--weight_decay_rate`: (weight decay rate)
  * `--learning_policy`: (learning rate decay policy, such as "plateau")
  * `--warm_start_from`: "004_triplet/best_model.ckpt-215001" (absolute or relative path)

* Other Parameters:
  * Please run `python ./entry/main.py --help` for details

**C. Add execution permission for your bash file**

```bash
chmod u+x ./run_scripts/my_config.sh
```

**D. Begin train/evaluate model**

* training
```bash
./run_scripts/my_config.sh train 0
./run_scripts/my_config.sh train 0,1   # Using multiple GPUs
```
* evaluation
```bash
# evaluation with best checkpoint
./run_scripts/my_config.sh eval 0 --load_status_file checkpoint_best
# evaluation with final checkpoint
./run_scripts/my_config.sh eval 0 --eval_final
# evaluating and saving predictions
./run_scripts/my_config.sh eval 0 --save_predict
```
* other parameters...

### 3. Train/Evaluate GUNet model

**A. For help**
```bash
python ./entry/main_g.py --help
```

**B. Write your bash file in `./run_scripts/my_config_v2.sh`.**

You can copy and modify provided template bash file in `./run_scripts/template/001_gunet.sh`.

* Required Parameters: The same as UNet.

* Optional Parameters: The same as UNet.

* Some Extra Parameters:
  * `--use_context`: Use context guide
  * `--context_list`: Paired context information: (feature name, feature length). For example: hist, 200
  * `--hist_noise`: Add noise to histogram context
  * `--hist_noise_scale`: Histogram noise random scale
  * `--use_spatial`: Use spatial guide
  * `--spatial_random`: Probability of adding spatial guide to current slice with tumors when use_spatial is on
  * `--spatial_inner_random`: Random choice tumors to give spatial guide inside a slice with tumors
  * `--save_sp_guide`: Save spatial guide when evaluating

Then add execution permission for your bash file & begin train/evaluate model the same as UNet.

### 4. Deployment

* Use CPU

```bash
docker run -t --rm -p 8501:8501 -v "D:/Library/Documents/0WorkSpace/MedicalImageSegmentation/export_path/112_nf_sp_dp:/models/112_nf_sp_dp" -e MODEL_NAME=112_nf_sp_dp tensorflow/serving &
```

* Use GPU (require docker `nvidia/cuda`)
```bash
docker run --runtime=nvidia -t --rm -p 8500:8500 -p 8501:8501 -v "/media/Data/Library/Documents/0WorkSpace/MedicalImageSegmentation/export_path/115_nf_both1_v2:/models/115_nf_both1_v2" -e MODEL_NAME=115_nf_both1_v2 tensorflow/serving:latest-gpu &
```

*   Use GPU (with custom driver installation)

```bash
docker run --runtime=nvidia -t --rm  -p 8500:8500 -p 8501:8501 -v "/usr/lib/nvidia-396:/custom/nvidia-396" -v "/home/mgh3dqi/zjw/MIS/export_path/112_nf_sp_fix:/models/112_nf_sp_fix" -e LD_LIBRARY_PATH=/custom/nvidia-396 -e NVIDIA_VISIBLE_DEVICES=1 -e MODEL_NAME=112_nf_sp_fix tensorflow/serving:latest-gpu-cuda9.0
```



## Need implemented

### Base

- [x] Data loader
- [x] Training routine
- [x] Summaries
- [x] Metrics
- [x] Loss functions
- [x] Evaluate
- [x] Predict
- [x] Visualization
- [x] Multi-GPU Training
- [ ] Multi-GPU Evaluation/Prediction

### Networks

- [x] UNet
- [ ] Attention UNet
- [x] GUNet
- [ ] G-AttentionUNet
- [ ] DenseUNet
- [ ] G-DenseUNet
- [ ] Deep Supervised UNet

- [ ] Pre-trained backbone
- [x] Context guide
- [x] Histogram context
- [ ] Other context
- [x] Spatial guide
