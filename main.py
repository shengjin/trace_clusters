#!/usr/bin/env python2

import numpy as np
from constant import *
import partition as pt
import clusters as clt

import time


# the critical distance used for partition clusters.
crit_d = 10.0
# the last 'space_dimension' column of the data are used to calc distances.
space_dimension = 3

accurate = False # using the grid-based FOF algorithm
recursive = True # using recursive fof finder in grid-based FOF

accurate = True  # using the FOF algorithm

# number of frames
n_frames = 2

# dt between two time frames
dt = 1

# write all these seperate clusters (inside the cluster class)
save_files = True

clusters_all = []
for i in range(n_frames):

    file_dir  = i+1

    filename = "%s%d%s" % ("./frame", file_dir, ".csv")
    points = np.genfromtxt(filename, skip_header=1, dtype=float, delimiter=',')

    if accurate:
        points_cluster = pt.infect_fof(points, crit_d, space_dimension, file_dir) 
    else:
        points_cluster = pt.infect_grid_fof(points, crit_d, space_dimension, file_dir, recursive) 

    clusters_new = clt.create_clusters(points_cluster, space_dimension, dt, save_files, file_dir)
    clusters_all.append(clusters_new)


#######################
# tracing 

delta_x = 0.2
delta_Var = 0.2
delta_J = 0.2

clt.trace_clusters(clusters_all[1], clusters_all[0], delta_x, delta_Var, delta_J)

print clusters_all[0]
print clusters_all[1]

