#!/usr/bin/env python

# Imports
from darkflow.net.build import TFNet
import tensorflow as tf


class DarkflowModel:
    def __init__(self,
                 model_config,
                 threshold: float,
                 labels='./labels.txt',
                 gpu_usage: float = 0.5,
                 weights=None,
                 model_pb=None,
                 model_meta=None,
                 checkpoint=None,
                 construct: bool = True):
        self.print_this("BEGIN INITIALIZATION")
        self.model_config = model_config
        self.labels = labels
        self.id_dict = self.load_id_dict(self.labels)
        self.model_pb = model_pb
        self.model_meta = model_meta
        self.use_pb_load = False
        if model_pb is not None:
            weights_info = "Loading from protobuf file: " + str(self.model_pb)
            self.use_pb_load = True
        elif weights is not None:
            self.load = weights
            weights_info = "Loading from weights file: " + str(self.load)
        elif checkpoint is not None:
            self.load = checkpoint
            weights_info = "Loading from checkpoint: " + str(self.load)
        self.threshold = threshold
        self.gpu_usage = gpu_usage
        self.model = None
        if construct:
            self.construct()
        self.print_this("Loading from config file: " + str(self.model_config))
        self.print_this("Loading labels from: " + str(self.labels))
        self.print_this("Detection Threshold: " + str(self.threshold))
        self.print_this("GPU Usage: " + str(self.gpu_usage))
        self.print_this(weights_info)
        self.print_this("INITIALIZATION COMPLETE")

    def load_id_dict(self, labels):
        id_dict = {}
        with open(labels, 'r') as lf:
            for num, line in enumerate(lf, 1):
                id_dict[line.strip()] = (num-1)
        return id_dict

    def construct(self):
        self.print_this("Constructing network")
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if self.use_pb_load:
                self.print_this("from protobuf")
                options = {
                    'model': self.model_config,
                    'labels': self.labels,
                    'pbload': self.model_pb,
                    'metaload': self.model_pb,
                    'threshold': self.threshold,
                    'gpu': self.gpu_usage
                }
            else:
                options = {
                    'model': self.model_config,
                    'labels':self.labels,
                    'load': self.load,
                    'threshold': self.threshold,
                    'gpu': self.gpu_usage
                }
        self.model = TFNet(options)
        self.print_this("Network Contructed")

    def infer_one(self, im):
        self.print_this("Inferring on image... ")
        result = self.model.return_predict(im)
        self.print_this("Inference complete!")
        return result

    def print_this(self, to_print):
        print("[DARKFLOW-MODEL]: ", end="")
        print(to_print)


