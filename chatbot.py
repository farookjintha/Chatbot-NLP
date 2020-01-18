# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:52:10 2020

@author: farookjintha
"""

import numpy as np
import tensorflow as tf
import re
import time


#Importing the dataset
lines = open(r"C:\Users\farookjintha\Desktop\Projects-FJ\FJ-chatbot\Chatbot-NLP\movie_lines.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open(r"C:\Users\farookjintha\Desktop\Projects-FJ\FJ-chatbot\Chatbot-NLP\movie_conversations.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]