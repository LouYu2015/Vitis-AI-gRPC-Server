from __future__ import print_function

import os
import re

import tensorflow as tf
import numpy as np
import cv2

from util import top5_accuracy
from vai.dpuv1.rt.xdnn_rt_tf import TFxdnnRT as xdnnRT
from vai.dpuv1.rt.xdnn_util import make_list
from vai.dpuv1.rt.xdnn_io import default_xdnn_arg_parser

QUANTIZE = True

# Environment Variables (obtained by running "source overlaybins/setup.sh")
HOME = os.getenv('HOME', '/home/mluser/')
VAI_ALVEO_ROOT = os.getenv('VAI_ALVEO_ROOT', os.getcwd() + '/../')

if os.path.isdir(os.path.join(VAI_ALVEO_ROOT, 'overlaybins', 'xdnnv3')):
    XCLBIN = os.path.join(VAI_ALVEO_ROOT, 'overlaybins', 'xdnnv3')
else:
    XCLBIN = os.path.join('/opt/xilinx', 'overlaybins', 'xdnnv3')

if 'VAI_ALVEO_ROOT' in os.environ and os.path.isdir(os.path.join(os.environ['VAI_ALVEO_ROOT'], 'vai/dpuv1')):
    ARCH_JSON = os.path.join(os.environ['VAI_ALVEO_ROOT'], 'vai/dpuv1/tools/compile/bin/arch.json')
elif 'CONDA_PREFIX' in os.environ and os.path.isdir(os.path.join(os.environ['CONDA_PREFIX'], 'arch')):
    ARCH_JSON = os.path.join(os.environ['CONDA_PREFIX'], 'arch/dpuv1/ALVEO/ALVEO.json')
else:
    ARCH_JSON = os.path.join(os.environ['VAI_ROOT'], 'compiler/arch/dpuv1/ALVEO/ALVEO.json')

MODELDIR = VAI_ALVEO_ROOT + "/examples/tensorflow/models/"
IMAGEDIR = HOME + "/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/"
IMAGELIST = HOME + "/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt"
LABELSLIST = HOME + "/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt"

print("Running w/ HOME: %s" % HOME)
print("Running w/ VAI_ALVEO_ROOT: %s" % VAI_ALVEO_ROOT)
print("Running w/ XCLBIN: %s" % XCLBIN)

quantInfo = MODELDIR + 'quantization_fix_info.txt'


def selectModel(MODEL):
    global protoBuffer, inputNode, outputNode, inputShape, means, pre_process

    default_protoBuffer = {'resnet50': 'resnet50_baseline.pb', 'inception_v1': 'inception_v1_baseline.pb', 'pedestrian_attribute': 'pedestrian_attributes_recognition_quantizations.pb'}
    default_inputNode   = {'resnet50': 'data', 'inception_v1': 'data', 'pedestrian_attribute': 'data'}
    default_outputNode  = {'resnet50': 'prob', 'inception_v1': 'loss3_loss3', 'pedestrian_attribute': 'pred_upper,pred_lower,pred_gender,pred_hat,pred_bag,pred_handbag,pred_backpack'}
    default_inputShape  = {'resnet50': '224,224', 'inception_v1': '224,224', 'pedestrian_attribute': '224,128'}
    default_means       = {'resnet50': '104,117,124', 'inception_v1': '104,117,124', 'pedestrian_attribute': '104,117,124'}

    if MODEL == "custom":
        protoBuffer = None
        inputNode   = None
        outputNode  = None
        inputShape  = None
        means       = None
        pre_process = None
    else:
        protoBuffer = MODELDIR + default_protoBuffer[MODEL]
        inputNode   = default_inputNode[MODEL]
        outputNode  = default_outputNode[MODEL]
        inputShape  = default_inputShape[MODEL]
        means       = default_means[MODEL]
        pre_process = MODEL
selectModel('resnet50')

print("Running with protoBuffer:   %s" % protoBuffer)
print("Running with quantInfo:     %s" % quantInfo)
print("Running with inputNode:     %s" % inputNode)
print("Running with outputNode:    %s" % outputNode)
print("Running with inputShape:    %s" % inputShape)
print("Running with means:         %s" % means)

if QUANTIZE:
    import subprocess
    subprocess.check_call(["vai_q_tensorflow",
                           "inspect", "--input_frozen_graph", protoBuffer])

    subprocess.check_call([
        "vai_q_tensorflow", "quantize",
        "--input_frozen_graph", protoBuffer,
        "--input_nodes",        inputNode,
        "--output_nodes",       outputNode,
        "--input_shapes",       "?,"+str(inputShape) + ",3",
        "--output_dir",         MODELDIR,
        "--input_fn",           "util.input_fn_" + pre_process,
        "--method",             "1",
        "--calib_iter",         "100"])

    subprocess.check_call([
        "vai_c_tensorflow",
        "--frozen_pb",          MODELDIR + "/deploy_model.pb",
        "--arch",               ARCH_JSON,
        "--output_dir",         MODELDIR,
        "--net_name",           quantInfo,
        "--quant_info"])