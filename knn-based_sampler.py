#!/home/aakash17/ms-project/venv/py38_depoco/bin/python

#======================================+
#                                      |
#      High Performance Script for     |
#         KNN-Based Subsampling        |
#--------------------------------------+
#        ( Aakash Singh Bais )         |
#             15 Dec 2022              |
#                                      |
#======================================+


"""
NOTE : The script performs subsampling based upon the following described psuedo-code:
1. Get the nearest neighbours of all the points in a ball region around them. (radius of ball = R, R remains the same for all points in a dataset.)
2. Sort the pts based upon the number of nearest neighbours as calculated in the previous step.
3. Pick the top X % of pts from this sorted list, where X = % compression needed. X = 100 - Y, where Y = loss % needed.
"""

import os
import sys
from tqdm import tqdm as tqdm
import numpy as np
import random as rand

sys.path.append("/home/aakash17/ms-project/depoco/")
#import dialogs as dia
#import point_cloud_utils as pcu
from multiprocessing import Pool
import argparse
import scipy.spatial as ss

parser = argparse.ArgumentParser(description="Script for high performance space-based sampling.")
parser.add_argument("in_dir", type=str, help="Location of the input directory.")
parser.add_argument("out_dir", type=str, help="Location of the output directory.")
parser.add_argument("dataset", type=str, help="The dataset used (skitti / dales)")
parser.add_argument("loss_level", type=str, help="Loss level in percent (62.5, 72.5, 82.5, 92.5)")
parser.add_argument("-shuffle", "--s", dest="shuffle", type=str, help="Shuffle the points in the cloud before performing subsampling", default="True")
parser.add_argument("-idw_power", "--idwp", dest="idw_power", type=float, help="The inverse power of the distance to be applied as weight for idw based subsampling. Use 0 to turn off idw", default=0)
parser.add_argument("-mem_subd", "--subd", dest="mem_subd", type=int, help="Append operations to large arrays / lists takes time. So, we append to smaller arrays and then combine those arrays to reduce time cost. Specify the total number of subdivisions to make. (default = 10)", default=10)
parser.add_argument("-mode", "--m", dest="mode", type=str, help="The mode of operation. ('compress' = to generate compressed results, 'append' = to just append a weight as a scalar to eery pt in the cloud)", default="append")
#parser.add_argument("-verbose", "--v", action="store_true")

args = parser.parse_args()
in_dir = args.in_dir
out_dir = args.out_dir
dataset = args.dataset
loss_level = float(args.loss_level)
shuffle = args.shuffle
idwp = float(args.idw_power)
mem_subdivisions = args.mem_subd
mode = args.mode
#verbose = args.verbose

# Select the input directory : where the uncompressed clouds are stored
#files_list = os.listdir(in_dir)

pts_perc_needed = (100-loss_level)/100

ball_radius = { # in meter
    "dales":5,
    "skitti":10 #10
}

# Perform the compression of every file and store in the output directory
out_path_base = out_dir #"/DATA/aakash/paper-1/compressed-data/multi-file-training/2.space-based/dales/"

#def say(*args, end=""):
#    global verbose
#    print("".join([str(arg) for arg in args]), end=str(end))
#
#    return

def ptDensitySort(cloud, cloud_tree, ball_radius, pts_needed:int):
    global idwp, mem_subdivisions
    #sorted = []
    #pts_needed = int(len(cloud)/3)
    
    sorted = np.zeros(shape=(pts_needed,2))
    for pt_counter in tqdm(range(pts_needed)):
        #len(cloud_tree.query_ball_point(cloud[pt_counter][:3], ball_radius))
        pt = cloud[pt_counter, :3]
        
        #if idwp != 0 or idwp == 0:
        idxs = cloud_tree.query_ball_point(pt, ball_radius)
        #print(cloud_tree.data[idxs].shape)
        if idwp != 0:
            sum_idw = np.power(np.average(np.absolute(ss.distance.cdist([pt], cloud_tree.data[idxs].tolist(), metric="euclidean"))),-idwp)
            #print(power.shape)
            sum_idw = sum_idw[~(np.isinf(sum_idw))]
        else:
            sum_idw = np.average(np.absolute(ss.distance.cdist([pt], cloud_tree.data[idxs].tolist(), metric="euclidean")))
            #print(sum_idw)
        #print(sum_idw)
        # [np.array(element) for element in cloud_tree.data[idxs]]
        if not sum_idw:
            sum_idw = 0
        sorted[pt_counter] = [pt_counter, sum_idw]
            
        #elif idwp == 0:
        #    point_tree = ss.cKDTree([pt])
        #    idxs_len = cloud_tree.count_neighbors(point_tree, ball_radius)
        #    sorted.append([pt_counter, idxs_len])
    
    # Combine all arrays stored in memory
    sorted = np.array(sorted)

    # Perform sorting on the sorted (list, convert it to numpy array first)
    sorted = sorted[sorted[:,1].argsort()[::-1]]

    # Use this sorted array to sort the points in the cloud based on the number of nearest neighours
    sorted_cloud = []

    for element in sorted:
        #print(element, type(int(element[0]), type(element[1]))
        sorted_cloud.append(list(np.concatenate((cloud[int(element[0])], [element[1]]), axis=0)))

    return np.array(sorted_cloud)

def append_file(file:str):
    global in_dir, out_dir, mode, pts_perc_needed

    compressed_cloud = []
    # Read the file
    cloud = np.genfromtxt(os.path.join(in_dir, file), delimiter=",")

    # Shuffle rows (points) of the cloud in random order, so we do not get artifacts which are a function of the order in which the points were collected
    if shuffle=="True":   
        np.random.shuffle(cloud)
    #print(cloud.shape)
    
    # Generate KD Tree on the file
    cloud_tree = ss.cKDTree(cloud[:,:3])
    
    # Get sorted array of pts that have max nearest neighbours in ball region
    pts_needed = int(pts_perc_needed*len(cloud))
    
    #vecPtDensitySort = np.vectorise(ptDensitySort)
    sorted_cloud = ptDensitySort(cloud, cloud_tree, ball_radius=ball_radius[dataset], pts_needed=pts_needed)

    # Write these pts to file in output directory
    compressed_cloud = sorted_cloud #[:pts_needed,:]

    # Write compressed cloud to disk
    np.savetxt(os.path.join(out_dir, file), compressed_cloud, delimiter=",", fmt="%s")

    return compressed_cloud

def compress_file(file:str):
    global out_dir, in_dir, mode, loss_level, pts_perc_needed
    
    in_file_path = os.path.join(in_dir, file)
    out_file_path = os.path.join(out_dir, file)

    cloud = np.genfromtxt(in_file_path, delimiter=",")
    out_cloud = cloud[:int(pts_perc_needed*len(cloud)),:5]
    np.savetxt(out_file_path, out_cloud, delimiter=",", fmt="%s")

    return out_cloud

if __name__ == "__main__":
    if os.path.isdir(in_dir):
        files_list = os.listdir(in_dir)
    elif os.path.isfile(in_dir):
        files_list = [in_dir.split("/")[-1]]
        in_dir = "/".join(in_dir.split("/")[:-1])
    else:
        raise Exception("Input directory path is neither a directory nor a file. Please enter a new path. \nExiting.")

    print("Processing " + str(len(files_list)) + " files...")
    
    function = append_file if mode == "append" else compress_file

    with Pool() as p:
        r = list(tqdm(p.imap(function, files_list), total=len(files_list), colour="blue"))
