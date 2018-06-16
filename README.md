# Installation

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Python 3.6
*   cuda-9.0
*   Protobuf 3+
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow - 1.8
*   Cython

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu 16.04 using via apt-get:

``` bash
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
sudo pip install Cython
sudo pip install jupyter
sudo pip install matplotlib
```

Alternatively, users can install dependencies using pip:

``` bash
sudo pip install Cython
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```

## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the tensorflow/models/research/ directory:


``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:


``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file, replacing \`pwd\` with the absolute path of
tensorflow/models/research on your system.

# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
python object_detection/builders/model_builder_test.py
```

# Preparing Inputs

In tensorflow/models/research/ 
* Create images and annotations folder. Inside annotations folder create xmls folder. 
* Copy the images to tensorflow/models/research/images folder. 
* Copy the xml_files to tensorflow/models/research/annotations/xmls folder. 
* Create data folder in `tensorflow/models/research/`
* Copy the `clamp_label_map.pbtxt` to `tensorflow/models/research/data/`

Note: The name of files in images and xmls folder should be same. 


Tensorflow Object Detection API reads data using the TFRecord file format. 

## Generating the TFRecord files.

```bash
python object_detection/dataset_tools/create_clamp_tf_record.py \
    --label_map_path=object_detection/data/clamp_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`
```

You should end up with two TFRecord files named `clamp_train.record` and
`clamp_val.record` in the `tensorflow/models/research/` directory.

## Recommended Directory Structure for Training and Evaluation

Move the created `clamp_train.record`, `clamp_val.record` and `clamp_label_map.pbtxt` to `tensorflow/models/research/data` folder.

From the tensorflow/models/research/ directory,

```
+data
  -label_map file
  -train TFRecord file
  -eval TFRecord file
+models
  + model
    -pipeline config file
    +train
    +eval
    +graph
```

## Creating the pipeline config file
* The faster_rcnn_resnet_101_coco and other config files can be found [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).
* The corresponding pretrained weight in different datasets can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
* Create pretrained_weight folder in `research/object_detection/samples/`
* Move faster_rcnn_resnet_101_coco pretrained weight to `research/object_detection/samples/pretrained_weight`
* Move faster_rcnn_resnet_101_coco configuration file to `research/models/model`

### Changes to make in configuration file:

Fine_tune_checkpoint

  ```bash
  {YOUR_PATH}/models/research/object_detection/samples/pretrained_weight/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt
  ```
Train path 

  ```bash
  train_input_reader: {
  tf_record_input_reader {
    input_path: "{YOUR_PATH}/models/research/data/clamp_train.record"}
    label_map_path: "{YOUR_PATH}/models/research/data/clamp_label_map.pbtxt"}
  ```
Test path

```bash
  eval_input_reader: {
  tf_record_input_reader {
    input_path:"{YOUR_PATH}/models/research/data/clamp_val.record"}
  label_map_path:"{YOUR_PATH}/models/research/data/clamp_label_map.pbtxt"} 
```

## Running the Training Job

A local training job can be run with the following command:

```bash
# From the tensorflow/models/research/ directory
python object_detection/train.py \
  --logtostderr \
  --pipeline_config_path = ${PATH_TO_YOUR_PIPELINE_CONFIG}
  --train_dir = ${PATH_TO_TRAIN_DIR}
```

where `${PATH_TO_YOUR_PIPELINE_CONFIG}` points to the pipeline config and
`${PATH_TO_TRAIN_DIR}` points to the directory in which training checkpoints
and events will be written to. By default, the training job will
run indefinitely until the user kills it.

## Running Tensorboard

Progress for training and eval jobs can be inspected using Tensorboard. If
using the recommended directory structure, Tensorboard can be run using the
following command:

```bash
tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}
```

where `${PATH_TO_MODEL_DIRECTORY}` points to the directory that contains the
train and eval directories. Please note it may take Tensorboard a couple minutes
to populate with data.


# Exporting a trained model for inference

After your model has been trained, you should export it to a Tensorflow
graph proto. A checkpoint will typically consist of three files:

* model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001
* model.ckpt-${CHECKPOINT_NUMBER}.index
* model.ckpt-${CHECKPOINT_NUMBER}.meta

After you've identified a candidate checkpoint to export, run the following
command from tensorflow/models/research:


```bash
# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory ${EXPORT_DIR}
```
${EXPORT_DIR} -- `models/research/models/model/graph`

Afterwards, you should see the directory ${EXPORT_DIR} containing the following:

* output_inference_graph.pb, the frozen graph format of the exported model
* saved_model/, a directory containing the saved model format of the exported model
* model.ckpt.*, the model checkpoints used for exporting
* checkpoint, a file specifying to restore included checkpoint files

# Infer the model with images in a given folder

```bash
python infer.py --model_file ${PATH TO MODEL FILE}\
      --input_path ${FOLDER PATH CONTAINING IMAGES}\
      --inp_img_ext ${INPUT IMAGE EXTENSION} \
      --output_path ${OUTPUT FOLDER PATH} \
      --label_file ${LABEL FILE PATH}
```

${PATH TO MODEL FILE} -- `{YOUR_PATH}/models/research/models/model/graph/frozen_inference_graph.pb`  
${FOLDER PATH CONTAINING IMAGES} -- Input folder which contains the evaluation images\
${INPUT IMAGE EXTENSION} - Image extension example -- bmp,jpg,png,..\
${OUTPUT FOLDER PATH} - Folder to which the images are saved \
${LABEL FILE PATH} - Path to label file `{YOUR_PATH}/models/research/data/clamp_label_map.pbtxt`


# Infer the model with video with a ip camera

 ```bash
python infer_video.py --model_file ${PATH TO MODEL FILE} --input_url ${URL} --label_file ${LABEL FILE PATH}
 ```
${PATH TO MODEL FILE} -- `{YOUR_PATH}/models/research/models/model/graph/frozen_inference_graph.pb`  
${LABEL FILE PATH} - `{YOUR_PATH}/models/research/data/clamp_label_map.pbtxt`