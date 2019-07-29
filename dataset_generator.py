#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:19:00 2019

@author: dayvsonsales
"""

import csv

def xor(x1, x2):
   return int((bool(x1) ^ bool(x2)) == True)

with open('/Users/dayvsonsales/trab-icomp/dataset_xor2.csv', mode='w') as file:
    fieldnames = ['x1', 'x2', 'x3', 'x4', 'result']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for x1 in range(2):
        for x2 in range(2):
          writer.writerow({'x1': x1, 'x2': x2, 'result': xor(x1, x2)})
