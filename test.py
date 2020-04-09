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

QUANTIZE = False

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
else:

    def get_args(startnode=inputNode, finalnode=outputNode):
        return {
            ### Some standard partitioner arguments [EDITABLE]
            'startnode': startnode,
            'finalnode': finalnode,

            ### Some standard compiler arguments [PLEASE DONT TOUCH]
            'dsp': 96,
            'memory': 9,
            'bytesperpixels': 1,
            'ddr': 256,
            'data_format': 'NHWC',
            'mixmemorystrategy': True,
            'noreplication': True,
            'xdnnv3': True,
            'usedeephi': True,
            'quantz': ''
        }


    ## load default arguments
    FLAGS, unparsed = default_xdnn_arg_parser().parse_known_args([])

    ### Partition and compile
    rt = xdnnRT(FLAGS,
                networkfile=protoBuffer,
                quant_cfgfile=quantInfo,
                xclbin=XCLBIN,
                device='FPGA',
                placeholdershape="{{'{}':[1,{},{},3]}}".format(inputNode, *[int(x) for x in inputShape.split(',')]),
                **get_args(inputNode, outputNode)
                )


    ## Pre-processing function
    def preprocess(image):
        input_height, input_width = 224, 224

        ## Image preprocessing using numpy
        img = cv2.imread(image).astype(np.float32)
        img -= np.array(make_list(means)).reshape(-1, 3).astype(np.float32)
        img = cv2.resize(img, (input_width, input_height))

        return img


    ## Choose image to run, display it for reference
    image = IMAGEDIR + "ILSVRC2012_val_00000003.JPEG"

    ## Accelerated execution

    ## load the accelerated graph
    graph = rt.load_partitioned_graph()

    ## run the tensorflow graph as usual (additional operations can be added to the graph)
    with tf.Session(graph=graph) as sess:
        input_tensor = graph.get_operation_by_name(inputNode).outputs[0]
        output_tensor = graph.get_operation_by_name(outputNode).outputs[0]

        import time
        for i in range(20):
            start_time = time.time()
            predictions = sess.run(output_tensor, feed_dict={input_tensor: [preprocess(image)]})
            end_time = time.time()
            print("time with preprocess:", end_time - start_time)
        preprocessed_image = preprocess(image)
        for i in range(20):
            start_time = time.time()
            predictions = sess.run(output_tensor, feed_dict={input_tensor: [preprocessed_image]})
            end_time = time.time()
            print("time without preprocess:", end_time - start_time)
        for i in range(20):
            start_time = time.time()
            predictions = sess.run(output_tensor, feed_dict={input_tensor: [preprocessed_image]*10})
            end_time = time.time()
            print("time without preprocess 10 batch:", end_time - start_time)
        for i in range(20):
            start_time = time.time()
            predictions = sess.run(output_tensor, feed_dict={input_tensor: [preprocessed_image]*100})
            end_time = time.time()
            print("time without preprocess 100 batch:", end_time - start_time)

    labels = np.loadtxt(LABELSLIST, str, delimiter='\t')
    top_k = predictions[0].argsort()[:-6:-1]

    for l, p in zip(labels[top_k], predictions[0][top_k]):
        print(l, " : ", p)

    iter_cnt = 100
    batch_size = 1
    label_offset = 0

    top5_accuracy(graph, inputNode, outputNode, iter_cnt, batch_size, pre_process, label_offset)
