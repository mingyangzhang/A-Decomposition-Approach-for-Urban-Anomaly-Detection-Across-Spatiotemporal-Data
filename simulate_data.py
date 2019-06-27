import pickle
import random
import scipy.io as sio
import numpy as np
import pandas as pd

from location_embedding import geo_embed
from utils import *

def generate_ny_flow(weight=2):
    feature = sio.loadmat("data/feature.mat")["feature"]
    flow = sio.loadmat("data/flow.mat")["flow"]
    week_num = 1
    repeat_num = 20
    # weather*5;weekday*7;hour*24
    normal_x = feature[:week_num*7*24, :, :]
    normal_y = np.zeros((week_num*7*24, 82, 4))
    for i in range(50):
        normal_y += flow[i*week_num*7*24:(i+1)*week_num*7*24, :, :]
    normal_y = normal_y/50

    print("Geo embedding starts!")
    geo_embed_size = 16
    hour_num, zone_num, _ = normal_y.shape
    for i in range(12):
        gfeature = geo_embed(normal_y[i::12, :, :], geo_embed_size)
        normal_x[i::12, :, 36:] = np.copy(gfeature)
    print("Geo embedding done.")

    normal_y = np.concatenate(repeat_num*[normal_y], axis=0)
    normal_x = np.concatenate(repeat_num*[normal_x], axis=0)

    n, m, _ = normal_x.shape

    means = np.mean(np.mean(normal_y, axis=0), axis=1)
    max_index = np.argmax(means)
    miss_index = np.where(means<50)[0]
    miss_len = miss_index.size
    for mi in miss_index:
        normal_x[:, mi, :] = normal_x[:, max_index, :]
        normal_y[:, mi, :] = normal_y[:, max_index, :]

    normal_y = normal_y * np.random.normal(1, 0.1, normal_y.shape)

    # weekends
    for i in range(24, n, 7*24):
        normal_y[i:i+24] = 0.6*normal_y[i:i+24]
        normal_y[i+24:i+48] = 0.5*normal_y[i+24:i+48]

    # # weather
    # # 2*snow
    rnd_days = range(0, week_num*7*repeat_num, 7)
    for day in rnd_days:
        hour = np.random.randint(0, 24)
        start = day*24 + hour
        end = day*24 + hour + 24
        # print("snow: {} - {}".format(start, end))
        normal_x[start:end, :, 24:24+5] = np.array([1, 0, 0, 0, 0])
        normal_y[start:end, :] = 0.5*normal_y[start:end, :]

    # # 14*rain
    rnd_days = range(2, week_num*7*repeat_num, 7)
    for day in rnd_days:
        hour = np.random.randint(0, 24)
        start = day*24 + hour
        end = day*24 + hour + 24
        normal_x[start:end, :, 24:24+5] = np.array([0, 0, 0, 0, 1])
        normal_y[start:end, :] = 0.6*normal_y[start:end, :]

    labels = np.zeros((n, m))

    # anomalies
    last_hour = 1
    test_week_num = 1
    effect_area = 1
    rnd_hours = np.random.randint(n-test_week_num*7*24, n, size=25)
    indexs = []
    for hour in rnd_hours:
        rnd_areas = np.random.randint(0, m-effect_area, size=2)
        for area in rnd_areas:
            scale = weight
            normal_y[hour:hour+last_hour, area:area+effect_area, :] = scale*normal_y[hour:hour+last_hour, area:area+effect_area, :]
            labels[hour:hour+last_hour, area:area+effect_area] = 1
            indexs.append((hour, hour+last_hour, area, effect_area))

    num_train_ano = int((n-test_week_num*7*24)/10/2)
    rnd_hours = np.random.randint(0, n-test_week_num*7*24, size=num_train_ano)
    for hour in rnd_hours:
        rnd_areas = np.random.randint(0, m-effect_area, size=2)
        for area in rnd_areas:
            scale = weight
            normal_y[hour:hour+last_hour, area:area+effect_area, :] = scale*normal_y[hour:hour+last_hour, area:area+effect_area, :]

    sio.savemat("data/fake_data.mat", {"x": normal_x, "y": normal_y, "index": indexs, "label": labels})
    print("Finish writing ny data.")
