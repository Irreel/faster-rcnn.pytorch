# Object Detection with Proposal Box Visualization

## Setup

### Dataset
download PASVAL VOC 2007 into `data/VOCdevkit/`

### Pretrained Model
download pretrained ResNet101 into `data/pretrained_model`

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
python setup.py build develop
```

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

To train a faster R-CNN model with resnet101 on pascal_voc, run:
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                    --dataset pascal_voc --net res101 \
                    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                    --cuda --use_tfb
```
or just simply `python trainval_net.py --use_tfb`


## Test

If you want to evlauate the detection performance of a pre-trained resnet101 model on pascal_voc test set, simply run
```
python test_net.py --dataset pascal_voc --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=20, CHECKPOINT=2504.

## Visualizing proposal boxes
Visualize the **proposal boxes** on four testing images.
Please download the pretrained model listed above or train your own models at first, then add images to folder `$ROOT/images`, and then run
```
python demo.py --net resnet101 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir path/to/model/directoy
```
The results is saved in `$ROOT/images/proposal_box`

## Demo
If you want to run detection on your own images with a pre-trained model, download the pretrained model listed above or train your own models at first, then add images to folder $ROOT/images, and then run
```
python demo.py --net resnet101 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir path/to/model/directoy
```
Then you will find the detection results in folder `$ROOT/images/det` and corresponding proposal boxes are saved in `$ROOT/images/proposal_box`

## Results
PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align)

model    | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mAP
---------|-----|--------|-----|-----|-------|-----
[Res-101](https://www.dropbox.com/s/4v3or0054kzl19q/faster_rcnn_1_7_10021.pth?dl=0)   | 1 | 1e-3 | 5   | 7   |  0.88 hr | 75.2
[Res-101](https://www.dropbox.com/s/8bhldrds3mf0yuj/faster_rcnn_1_10_2504.pth?dl=0)   | 4 | 4e-3 | 8   | 10  |  0.60 hr | 74.9
[Res-101](https://www.dropbox.com/s/5is50y01m1l9hbu/faster_rcnn_1_10_625.pth?dl=0)    | 16| 1e-2 | 8   | 10  |  0.23 hr | 75.2 
[Res-101](https://www.dropbox.com/s/cn8gneumg4gjo9i/faster_rcnn_1_12_416.pth?dl=0)    | 24| 1e-2 | 10  | 12  |  0.17 hr | 75.1  

## Credits
[jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)