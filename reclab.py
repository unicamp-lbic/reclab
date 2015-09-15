# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:54:46 2015

@author: thalita
"""

import argparse
import pickle as pkl
import config

parser = argparse.ArgumentParser(description='Run recommender training/ evaluation')
parser.add_argument('config', help='config for this run')
args = parser.parse_args()

conf = config.config_dict[args.config]

