
import numpy as np
from constant import *
import os

##########################################
#  Algorithm 1, using distance
##########################################

############ Functions

def calc_distance(p_one, points_all, space_dimension):
    """
    To calculate the the distances of all the points regarding p_one.
    the last 'n_omit' elements of each points store non_coordinate infos
    """
    distance_square =  (points_all[:,-space_dimension-1:-1] - p_one[-space_dimension-1:-1])**2.0
    return (np.sum(distance_square, axis=1))**0.5


def infect_fof(points, crit_d, space_dimension, file_dir):
    """
    cluster partition, then add a column to indicate the sequence number 
    of the cluster at the end of each row.
    """

    # save the which nCluster each point belong to (at infect function)
    save_nCluster = False

    file_dir = str("%04d" % file_dir)
    
    npoints = points.shape[0]
    data_dimension = points.shape[1]
    
    points_cluster = np.zeros((npoints,data_dimension+1), dtype=float)
    for i in range(data_dimension):
        points_cluster[:,i] = points[:,i]
    
    # the n_th index cluster
    n_cluster = 0
    for i in range(npoints):
        # print out the progress
        if(i%max(1,int(0.1*npoints))==0):
            print "%s%s" % (100*i/npoints, "% finished ...")
        if (points_cluster[i,-1] == 0):
            # for each point, calc the distances to other points
            p_ref = points_cluster[i,:]
            distance = calc_distance(p_ref, points_cluster, space_dimension)
            # find all the points that have disctance < crit_d
            ind = np.where( distance < crit_d )[0]
            # points have distance < crit_d and with n_cluster > 0
            ind_nonzero = np.where( points_cluster[ind,-1] > 0)[0]
            if len(ind_nonzero) == 0:
                # then give these points a new n_cluster
                n_cluster = n_cluster + 1
                points_cluster[ind,-1] = n_cluster 
            else:
                # what clusters are these points from
                class_exist=np.unique(points_cluster[ind[ind_nonzero],-1], return_counts=1)
                if class_exist[0].shape[0]==1:
                    # only one cluster, set all the points to this cluster
                    ind_zero = np.where( points_cluster[ind,-1] == 0)[0]
                    points_cluster[ind[ind_zero],-1] = class_exist[0][0]
                else:
                    # merge these clusters into one
                    for j in range(class_exist[0].shape[0]-1):
                        points_cluster[np.where(points_cluster[:,-1]==class_exist[0][j+1]),-1] = class_exist[0][0]
                    # asign new points to this cluster
                    ind_zero = np.where( points_cluster[ind,-1] == 0)[0]
                    points_cluster[ind[ind_zero],-1] = class_exist[0][0]
    
    # save the cluster ID for all points for debug
    if save_nCluster:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir, 0755)
        name = "%s%s%s%s" % ("./", file_dir, "/", "clusterN.txt")
        np.savetxt(name, np.transpose([points_cluster[:,data_dimension]]), fmt='%.1f')
    
    return points_cluster


##########################################
#  Algorithm 2, using grid
# NOTE: faster, but not accurate
##########################################

########   Functions

def friend(x_n, n_cells, neighbor):
    # find the friends of one cell
    # NOTE: i_cell starts from 1, not 0.
    i = x_n[0]
    j = x_n[1]
    k = x_n[2]
    ninj = n_cells[0]*n_cells[1]
    ni = n_cells[0]
    #
    friends = np.zeros((26,3), dtype=float)
    friends_id = np.zeros(26)
    #
    friends[:,:] = [i,j,k]
    friends = friends + neighbor
    #
    friend_id = ninj*(friends[:,2]-1.0)+ni*(friends[:,1]-1.0)+friends[:,0]
    #
    final_id = []
    for l in range(26):
        if  (friends[l,:].prod() != 0) and (friends[l,0] <= n_cells[0]) and (friends[l,1] <= n_cells[1]) and (friends[l,2] <= n_cells[2]):
            final_id.append(friend_id[l])
    return final_id
    

# recursive fof finder
def recursive_fof(friend_all, mini_clusters, n, n_cells):
    # NOTE: need check!
    # find friends of current cell
    friend_once = friend(mini_clusters[n,:], n_cells, neighbor)
    for i in range(len(friend_once)):
        if (friend_once[i] in mini_clusters[:,-1]) and (friend_once[i] not in friend_all):
            friend_all.append(friend_once[i])
            nn = np.where(mini_clusters[:,-1] == friend_once[i])[0][0]
            recursive_fof(friend_all, mini_clusters, nn, n_cells)
    return  friend_all

# iteration fof finder
def iteration_fof(friend_all, mini_clusters, n, n_cells):
    # NOTE: need check!
    # find friends of current cell
    friend_once = friend(mini_clusters[n,:], n_cells, neighbor)
    while len(friend_once) > 0:
        if (friend_once[0] in mini_clusters[:,-1]) and (friend_once[0] not in friend_all):
            friend_all.append(friend_once[0])
            nn = np.where(mini_clusters[:,-1] == friend_once[0])[0][0]
            friend_new = friend(mini_clusters[nn,:], n_cells, neighbor)
            for i in range(len(friend_new)):
                if (friend_new[i] in mini_clusters[:,-1]) and (friend_new[0] not in friend_all):
                    friend_once.append(friend_new[i])
            friend_once.remove(friend_once[0])
        else:
            friend_once.remove(friend_once[0])
    return  friend_all


def merge_remove(points_cluster, friend_all, mini_clusters):
    for i in range(len(friend_all)-1):
        i = i+1
        ind = np.where(points_cluster[:,-1] == friend_all[i])[0]
        points_cluster[ind,-1] = friend_all[0]
        nn = np.where(mini_clusters[:,-1] == friend_all[i])[0][0]
        mini_clusters[nn,:] = 0.0
    nn  = np.where(mini_clusters[:,-1] == friend_all[0])[0][0]
    mini_clusters[nn,:] = 0.0
    return mini_clusters, points_cluster


# a new approach
def infect_grid_fof(points, crit_d, space_dimension, file_dir, recursive):

    save_nCluster = False
    file_dir = str("%04d" % file_dir)
    
    npoints = points.shape[0]
    data_dimension = points.shape[1]

    points_cluster = np.zeros((npoints,data_dimension+1), dtype=float)
    for i in range(data_dimension):
        points_cluster[:,i] = points[:,i]
    
    # length of each cubic cell
    d_cell = ((3.0*crit_d)**2.0/(4.0*space_dimension))**0.5
    #d_cell = (crit_d**2.0/(4.0*space_dimension))**0.5
    
    # how many cell in each space dimension
    n_cell = np.zeros(space_dimension)
    for i in range(space_dimension):
        n_cell[i] = (points[:,15+i].max()-points[:,15+i].min())//d_cell+1
    print n_cell

    xyz_cluster = np.zeros((npoints,space_dimension+1), dtype=float)
    # numbering these cells, i > x dimension; j > y; k > z.
    xyz_cluster[:,0] = (points[:,15]-points[:,15].min())//d_cell + 1.0
    xyz_cluster[:,1] = (points[:,16]-points[:,16].min())//d_cell + 1.0
    xyz_cluster[:,2] = (points[:,17]-points[:,17].min())//d_cell + 1.0
    xyz_cluster[:,-1] = (n_cell[0]*n_cell[1])*(xyz_cluster[:,2]-1.0)  \
                            + n_cell[0]*(xyz_cluster[:,1]-1.0) + xyz_cluster[:,0]
    points_cluster[:,-1] = xyz_cluster[:,-1]

    total_mini_clusters, indices = np.unique(xyz_cluster[:,-1], return_index=True)
    n_mini_clusters = total_mini_clusters.shape[0]
    mini_clusters = xyz_cluster[indices,:]

    for i in range(n_mini_clusters):
        # print out the progress
        if(i%max(1,int(0.1*n_mini_clusters))==0):
            print "%s%s" % (100*i/n_mini_clusters, "% finished ...")
        if mini_clusters[i,-1] != 0.0:
            # add the current cell
            friend_all = []
            friend_all.append(mini_clusters[i,-1])
            # the numbering starts from 1, not 0
            if recursive:
                # tailrecursive_fof not work yet
                # tailrecursive_fof(friend_all, mini_clusters, i, n_cell)
                recursive_fof(friend_all, mini_clusters, i, n_cell)
            else:
                iteration_fof(friend_all, mini_clusters, i, n_cell)

            merge_remove(points_cluster, friend_all, mini_clusters)

    # save the cluster ID for all points for debug
    if save_nCluster:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir, 0755)
        name = "%s%s%s%s" % ("./", file_dir, "/", "clusterN.txt")
        np.savetxt(name, np.transpose([points_cluster[:,data_dimension]]), fmt='%.1f')
    
    return points_cluster


