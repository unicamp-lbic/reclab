# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:54:28 2015

@author: thalita

Config file
"""
from colections import namedtuple
import recommenders as rec

import time

config_dict = {
    'DefaultConfig': DefaultConfig
}

def default_id():
    return time.strftime('%Y%m%d%H%M%S')

Config = namedtuple ('Config',
     ['experiment_id','result_folder', 'database_folder', 'RS_type', 'RS_args']
    )

DefaultConfig = Config(
    experiment_id=default_id(),
    result_folder='results/' + experiment_id + '/'),
    database=ml100k,
    RS_type=rec.ItemBased,
    RS_args={}
    )