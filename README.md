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
  * --tag (better to use current script name, such as ${BASE_NAME%".sh"})
  * --model {UNet}
  * --classes (your class names, such as "Liver")
  * --dataset_for_train (a list of *.tfrecord files or *.json file)
  * --dataset_for_eval_while_train (a list of *.tfrecord files or *.json file)

* Optional Parameters:
  * --zoom --noise --flip --zoom_scale 1.2 (for data augmentation)
  * --im_height 256 --im_width 256 (specify training image size, -1 if not specified)
  * --im_channel 1 (specify image channel)
  * --resize_for_batch (resize training image to the same size for batching operation)
  * --num_of_total_steps (number of max training steps)
  * --primary_metric ("<class>/<metric>", such as "Liver/Dice")
  * --weight_decay_rate (weight decay rate)
  * --learning_policy (learning policy, such as "custom_step")
  * --lr_decay_boundaries (in which step to decay learning rate, such as "200000 300000")
  * --lr_custom_values (learning rate values in each interval, such as "0.003 0.0003 0.0001")
  * --input_group (group image neighbor slices as a multi-channel input, such as "3". This
    parameter must be match with your examples in *.tfrecord dataset)
  * --warm_start_from "004_triplet/best_model.ckpt-215001" (absolute or relative path)

* Other Parameters:
  * Please run `python main.py --help` for details

**C. Add execution permission for your bash file: `chmod u+x ./run_scripts/train_unet.sh`**
* training
```bash
./run_scripts/train_unet.sh train 0
./run_scripts/train_unet.sh train 0,1   # Using multiple GPUs
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

### 3. How to train/eval a OSMNUNet model with your data ?

**A. Create `tumor_summary.csv` by running `./data_kits/analyze_lits.py`

**B. Mostly train/eval routine is the same with UNet, but some details are different. Please check
  `./run_scripts/016_osmn_in_noise.sh` for details. Notice that the entry becomes `main_osmn.py`.
  * For training with single gpu: `./run_scripts/xxx.sh train 0`
  * For training with multiple gpus: `./run_scripts/xxx.sh train 0,1`
  * For evaluating with single gpu: `./run_scripts/xxx.sh eval 0 --use_fewer_guide --guide middle`
  * For evaluating with multiple gpus: not supported!


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
- [x] OSMN

- [ ] Balance dataset
- [ ] Pre-trained backbone
- [x] Spatial guide
- [x] Gaussian distribution
