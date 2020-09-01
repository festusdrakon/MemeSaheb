# import cv2
# import re

# pat = r'/([^/]+)_\d+.jpg$'
# s1='/data1/jhoward/git/course-v3/nbs/dl1/data/oxford-iiit-pet/images/american_bulldog.jpg'
# x=re.search(pat, s1)
# #s=s1[x.start():]
# # print (s)

# regex = r'\\(?:.(?!\\))+$'
# path = 'D:\danklearning\images'
# image
# x=re.search(regex, path)
# print (x)

import pandas as pd
import numpy as np
import tqdm
from fastai import *
from fastai.vision import *
import torch

def func():
    image_path = 'C:/danklearning/images'
    filenames = get_image_files(image_path)
    np.random.seed(2)
    pat_reg_ex = r'-+.jpg$'
    data = ImageDataBunch.from_name_re(image_path, filenames, pat_reg_ex, ds_tfms=get_transforms(), size=224, bs=16)
    data.normalize(imagenet_stats)

    learn = ConvLearner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(4)
    learn.save('s1')

data = pd.read_csv('sample.txt',sep=" - ")
name = "forever alone"
df = data[data['label'].str.contains(name)]
df1 = df[['caption']]
df.to_csv(r'D:/danklearning/project\lstm.csv')


