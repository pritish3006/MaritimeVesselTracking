# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
import math


def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    scaler = StandardScaler()
    #trainFeatures = scaler.fit_transform(trainFeatures)
    testFeatures = scaler.fit_transform(testFeatures)
    km = KMeans(n_clusters=numVessels, random_state=100)
    predVessels = km.fit_predict(testFeatures)
    # sc = SpectralClustering(n_clusters=numVessels, gamma=1.0, affinity='rbf')
    # predVessels = sc.fit_predict(testFeatures)
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

def mile_conversion(latitude, longitude):
    mile_lat = latitude*69
    mile_long = longitude*54.6
    return [mile_lat, mile_long]


def euclidean_distance(location1, location2):
    x = abs(location1[0] - location2[0])**2
    y = abs(location1[1] - location2[1])**2
    return math.sqrt(x + y)

def predictive_movement(location, speed, course, time):
    radian_conversion = 360 - course*.1 + 90 
    y = location[0]
    x = location[1]
    y_change = y + math.sin(radian_conversion)*speed*abs(time)
    x_change = x + math.cos(radian_conversion)*speed*abs(time)
    #new latitude and longitude
    return [y_change, x_change]

#49 and 11


def cluster_amount(location, speed, time, course):
    locations = []
    speeds = []
    times = []
    courses = []
    i = 0
    while i < len(location):
        new_location = location[i]
        new_location = mile_conversion(new_location[0], new_location[1])
        if(len(locations) == 0):
            locations.append(new_location)
            speeds.append(speed[i])
            times.append(time[i])
            courses.append(course[i])
            i += 1
        else:
            found = False
            j = 0
            while j < len(locations):
                cluster_point = predictive_movement(locations[j], speeds[j]*.1, courses[j], ((time[i] - times[j])/3600))
                if(cluster_point[0]*.99 <= new_location[0] <= cluster_point[0]*1.01 and cluster_point[1]*.99 <= new_location[1] <= cluster_point[1]*1.01):
                    locations[j] = new_location
                    speeds[j] = speed[i]
                    times[j] = time[i]
                    courses[j] = course[i]
                    found = True
                    j = len(locations)
                else:
                    j += 1
            if(found == False):
                locations.append(new_location)
                speeds.append(speed[i])
                times.append(time[i])
                courses.append(course[i])
                i += 1
    return len(locations)






#%% 
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    #%%%
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]
    k = 0
    lat_long = []
    while k < len(features[:,1]):
        lat_long.append([features[k,1],features[k,2]])
        k += 1
    print(cluster_amount(lat_long, features[:,3], features[:,0],features[:,4]))
    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    
# %%
