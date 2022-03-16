# **maskrcnn-benchmark**

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models using PyTorch 1.0.

> This repository is finetuned to spearhead a quick way to set up and export .pth files to .onnx models.

## **Setup**
Follow the commands below to set up `maskrcnn_benchmark`.
```bash
cd $HOME
git clone https://github.com/cardboardcode/maskrcnn-benchmark.git --branch onnx_stage_mrcnn
cd maskrcnn-benchmark
conda create --name maskrcnn-benchmark python=3.6
conda activate maskrcnn-benchmark
pip install -r requirements.txt
python setup.py build develop
```

## **Verify**
Check that `maskrcnn_benchmark` has been successfully installed.
```bash
pip freeze | grep mask
# maskrcnn-benchmark==0.1
```

```bash
python demo/export_to_onnx.py
```
You should see a similar **output** as shown below:
```bash
/home/cardboardvoice/anaconda3/envs/maskrcnn-benchmark/lib/python3.6/site-packages/torch/nn/functional.py:2390: UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
  warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
demo/export_to_onnx.py:49: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])
/home/cardboardvoice/anaconda3/envs/maskrcnn-benchmark/lib/python3.6/site-packages/maskrcnn_benchmark-0.1-py3.6-linux-x86_64.egg/maskrcnn_benchmark/structures/bounding_box.py:28: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if bbox.size(-1) != 4:
/home/cardboardvoice/anaconda3/envs/maskrcnn-benchmark/lib/python3.6/site-packages/maskrcnn_benchmark-0.1-py3.6-linux-x86_64.egg/maskrcnn_benchmark/modeling/rpn/inference.py:105: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
...
2022-03-13 15:31:50.149424792 [W:onnxruntime:, graph.cc:1074 Graph] Initializer 6945 appears in graph inputs and will not be treated as constant value/weight. This may prevent some of the graph optimizations, like const folding. Move it out of graph inputs if there is no need to override it, by either re-generating the model with latest exporter/converter or with the tool onnxruntime/tools/python/remove_initializer_from_input.py.
Avg run time of onnx model: 0.16126501449980424
```
`mask_rcnn_R_50_FPN_1x.onnx` should be generated.

## Webcam and Jupyter notebook demo

We provide a simple webcam demo that illustrates how you can use `maskrcnn_benchmark` for inference:
```bash
cd demo
# by default, it runs on the GPU
# for best results, use min-image-size 800
python webcam.py --min-image-size 800
# can also run it on the CPU
python webcam.py --min-image-size 300 MODEL.DEVICE cpu
# or change the model that you want to use
python webcam.py --config-file ../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml --min-image-size 300 MODEL.DEVICE cpu
# in order to see the probability heatmaps, pass --show-mask-heatmaps
python webcam.py --min-image-size 300 --show-mask-heatmaps MODEL.DEVICE cpu
# for the keypoint demo
python webcam.py --config-file ../configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml --min-image-size 300 MODEL.DEVICE cpu
```

A notebook with the demo can be found in [demo/Mask_R-CNN_demo.ipynb](demo/Mask_R-CNN_demo.ipynb).

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.


## Model Zoo and Baselines

Pre-trained models, baselines and comparison with Detectron and mmdetection
can be found in [MODEL_ZOO.md](MODEL_ZOO.md)

## Inference in a few lines
We provide a helper class to simplify writing inference pipelines using pre-trained models.
Here is how we would do it. Run this from the `demo` folder:
```python
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = ...
predictions = coco_demo.run_on_opencv_image(image)
```

## Perform training on COCO dataset

For the following examples to work, you need to first install `maskrcnn_benchmark`.

You will also need to download the COCO dataset.
We recommend to symlink the path to the coco dataset to `datasets/` as follows

We use `minival` and `valminusminival` sets from [Detectron](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md#coco-minival-annotations)

```bash
# symlink the coco dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
# or use COCO 2017 version
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017

# for pascal voc dataset:
ln -s /path_to_VOCdevkit_dir datasets/voc
```

P.S. `COCO_2017_train` = `COCO_2014_train` + `valminusminival` , `COCO_2017_val` = `minival`


You can also configure your own paths to the datasets.
For that, all you need to do is to modify `maskrcnn_benchmark/config/paths_catalog.py` to
point to the location where your dataset is stored.
You can also create a new `paths_catalog.py` file which implements the same two classes,
and pass it as a config argument `PATHS_CATALOG` during training.

### Single GPU training

Most of the configuration files that we provide assume that we are running on 8 GPUs.
In order to be able to run it on fewer GPUs, there are a few possibilities:

**1. Run the following without modifications**

```bash
python /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "/path/to/config/file.yaml"
```
This should work out of the box and is very similar to what we should do for multi-GPU training.
But the drawback is that it will use much more GPU memory. The reason is that we set in the
configuration files a global batch size that is divided over the number of GPUs. So if we only
have a single GPU, this means that the batch size for that GPU will be 8x larger, which might lead
to out-of-memory errors.

If you have a lot of memory available, this is the easiest solution.

**2. Modify the cfg parameters**

If you experience out-of-memory errors, you can reduce the global batch size. But this means that
you'll also need to change the learning rate, the number of iterations and the learning rate schedule.

Here is an example for Mask R-CNN R-50 FPN with the 1x schedule:
```bash
python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
```
This follows the [scheduling rules from Detectron.](https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14-L30)
Note that we have multiplied the number of iterations by 8x (as well as the learning rate schedules),
and we have divided the learning rate by 8x.

We also changed the batch size during testing, but that is generally not necessary because testing
requires much less memory than training.

Furthermore, we set `MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000` as the proposals are selected for per the batch rather than per image in the default training. The value is calculated by **1000 x images-per-gpu**. Here we have 2 images per GPU, therefore we set the number as 1000 x 2 = 2000. If we have 8 images per GPU, the value should be set as 8000. Note that this does not apply if `MODEL.RPN.FPN_POST_NMS_PER_BATCH` is set to `False` during training. See [#672](https://github.com/facebookresearch/maskrcnn-benchmark/issues/672) for more details.

### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "path/to/config/file.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN images_per_gpu x 1000
```
Note we should set `MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN` follow the rule in Single-GPU training.

## Evaluation
You can test your model directly on single or multiple gpus. Here is an example for Mask R-CNN R-50 FPN with the 1x schedule on 8 GPUS:
```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/test_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" TEST.IMS_PER_BATCH 16
```
To calculate mAP for each class, you can simply modify a few lines in [coco_eval.py](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/datasets/evaluation/coco/coco_eval.py). See [#524](https://github.com/facebookresearch/maskrcnn-benchmark/issues/524#issuecomment-475118810) for more details.

## Abstractions
For more information on some of the main abstractions in our implementation, see [ABSTRACTIONS.md](ABSTRACTIONS.md).

## Adding your own dataset

This implementation adds support for COCO-style datasets.
But adding support for training on a new dataset can be done as follows:
```python
from maskrcnn_benchmark.structures.bounding_box import BoxList

class MyDataset(object):
    def __init__(self, ...):
        # as you would do normally

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = ...

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # and labels
        labels = torch.tensor([10, 20])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": img_height, "width": img_width}
```
That's it. You can also add extra fields to the boxlist, such as segmentation masks
(using `structures.segmentation_mask.SegmentationMask`), or even your own instance type.

For a full example of how the `COCODataset` is implemented, check [`maskrcnn_benchmark/data/datasets/coco.py`](maskrcnn_benchmark/data/datasets/coco.py).

Once you have created your dataset, it needs to be added in a couple of places:
- [`maskrcnn_benchmark/data/datasets/__init__.py`](maskrcnn_benchmark/data/datasets/__init__.py): add it to `__all__`
- [`maskrcnn_benchmark/config/paths_catalog.py`](maskrcnn_benchmark/config/paths_catalog.py): `DatasetCatalog.DATASETS` and corresponding `if` clause in `DatasetCatalog.get()`


## Finetuning from Detectron weights on custom datasets
Create a script `tools/trim_detectron_model.py` like [here](https://gist.github.com/wangg12/aea194aa6ab6a4de088f14ee193fd968).
You can decide which keys to be removed and which keys to be kept by modifying the script.

Then you can simply point the converted model path in the config file by changing `MODEL.WEIGHT`.

For further information, please refer to [#15](https://github.com/facebookresearch/maskrcnn-benchmark/issues/15).

## **Uninstall**
Follow the commands to completely remove `maskrcnn_benchmark`:
```bash
conda deactivate
conda env remove --name maskrcnn-benchmark
```
