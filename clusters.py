
import numpy as np
import os


##########################################*********************************************
class clusters():
    """
    class to store infos of all the clusters

    ATTRIBUTES:
    -----------
        size               - the number of clusters
        volume             - volumn ( the number of cells of each cluster
        x(space_dimension) - mean x_1, x_2, ..., x_space_dimension of each cluster
        E_x(space_dimension) - Expection of mean x_1, x_2, ..., x_space_dimension 
                                at next timeframe of each cluster
        Var_x(space_dimension) - Variances of  x_1, x_2, ..., x_space_dimension of each cluster
        Var_E_x(space_dimension) - Variances Expection of x_1, x_2, ..., x_space_dimension 
                                at next timeframe of each cluster
        v(space_dimension) - mean v_1, v_2, ..., v_space_dimension of each cluster
        absJ               - abs(J) of each cluster
        B                  - magnetic field: B0, B1, B3
        old_id             - id in previous frame (-1 means not found)
        prob_old_id        - probability of right answer
    """

    def __init__(self, size, dimension):
        self.size        = size
        self.volume      = np.zeros(size)
        self.x           = np.zeros((size,dimension), dtype=float)
        self.E_x         = np.zeros((size,dimension), dtype=float)
        self.Var_x       = np.zeros((size,dimension), dtype=float)
        self.Var_E_x     = np.zeros((size,dimension), dtype=float)
        self.v           = np.zeros((size,dimension), dtype=float)
        self.absJ        = np.zeros(size)
        self.B           = np.zeros((size,3), dtype=float)
        self.old_id      = np.zeros(size)
        self.prob_old_id = np.zeros(size)

    def __str__(self):
        s = "\n" + "cluster size:" + str(self.size) + "\n"
        s = s + "\n" + "self.x, self.v, self.E_x:" + "\n"
        for i in range(len(self.x)):
            s = s + "\n" + str(i+1) + "_th: " + str(self.x[i]) + "   " + \
                    str(self.E_x[i]) + "   " + str(self.v[i]) + "\n"
        s = s + "\n" + "self.Var_x, self.Var_E_x:" + "\n"
        for i in range(len(self.x)):
            s = s + "\n" + str(i+1) + "_th: " + str(self.Var_x[i,:]) + "   " + \
                    str(self.Var_E_x[i,:]) + "\n"
        s = s + "\n" + "self.volume, self.absJ, self.B[:], self.old_id, self_prob_old_id:" + "\n"
        for i in range(len(self.volume)):
            s = s + "\n" + str(i+1) + "_th: " + str(self.volume[i]) + "   " + str(self.absJ[i]) \
                    + "   " + str(self.B[i,:]) + "   " +  str(self.old_id[i]) \
                    + "   " +  str(self.prob_old_id[i]) + "\n"
        return s


######## Define a function to create the clusters using points_cluster
def create_clusters(points_cluster, space_dimension, dt, save_files, file_dir):

    file_dir = str("%04d" % file_dir)
    if save_files:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir, 0755)

    # re-order n_cluster in natural sequence
    # NOTE: n_cluster starts from 1, since 0 is used to 
    #       indicate the states in function infect
    cluster_name=np.sort(np.unique(points_cluster[:,-1], return_counts=1)[0][:])
    for i in range(cluster_name.shape[0]):
        points_cluster[np.where(points_cluster[:,-1]==cluster_name[i]),-1] = i+1

    size = cluster_name.shape[0]
    # create a cluster Class
    one_frame_clusters = clusters(size, space_dimension)

    # save the all clusters into seperate files for debug and plotting
    for i in range(cluster_name.shape[0]):
        one_cluster=np.asarray(points_cluster[np.where(points_cluster[:,-1]==(i+1)),:])
        if save_files:
            name = "%s%s%s%d%s" % ("./", file_dir, "/", i+1, ".txt")
            np.savetxt(name, one_cluster[0,:,:], fmt='%.5f')
        one_frame_clusters.volume[i] = one_cluster.shape[1]
        one_frame_clusters.absJ[i]   = np.average(one_cluster[0,:,0])
        one_frame_clusters.B[i,0] = np.average(one_cluster[0,:,3])
        one_frame_clusters.B[i,1] = np.average(one_cluster[0,:,4])
        one_frame_clusters.B[i,2] = np.average(one_cluster[0,:,5])
        one_frame_clusters.x[i,0] = np.average(one_cluster[0,:,15])
        one_frame_clusters.x[i,1] = np.average(one_cluster[0,:,16])
        one_frame_clusters.x[i,2] = np.average(one_cluster[0,:,17])
        one_frame_clusters.v[i,0] = np.average(one_cluster[0,:,12])
        one_frame_clusters.v[i,1] = np.average(one_cluster[0,:,13])
        one_frame_clusters.v[i,2] = np.average(one_cluster[0,:,14])
        one_frame_clusters.E_x[i,0] = np.average((one_cluster[0,:,15]+one_cluster[0,:,12]*dt))
        one_frame_clusters.E_x[i,1] = np.average((one_cluster[0,:,16]+one_cluster[0,:,13]*dt))
        one_frame_clusters.E_x[i,2] = np.average((one_cluster[0,:,17]+one_cluster[0,:,14]*dt))
        one_frame_clusters.Var_x[i,0] = np.average((one_cluster[0,:,15]-one_frame_clusters.x[i,0])**2.0)
        one_frame_clusters.Var_x[i,1] = np.average((one_cluster[0,:,16]-one_frame_clusters.x[i,1])**2.0)
        one_frame_clusters.Var_x[i,2] = np.average((one_cluster[0,:,17]-one_frame_clusters.x[i,2])**2.0)
        one_frame_clusters.Var_E_x[i,0] = np.average(((one_cluster[0,:,15]+one_cluster[0,:,12]*dt)-one_frame_clusters.E_x[i,0])**2.0)
        one_frame_clusters.Var_E_x[i,1] = np.average(((one_cluster[0,:,16]+one_cluster[0,:,13]*dt)-one_frame_clusters.E_x[i,1])**2.0)
        one_frame_clusters.Var_E_x[i,2] = np.average(((one_cluster[0,:,17]+one_cluster[0,:,14]*dt)-one_frame_clusters.E_x[i,2])**2.0)

    return one_frame_clusters


######## Function to trace cluster evolution
def trace_clusters(clusters_new, clusters_old, delta_x=0.1, delta_Var=0.1, delta_J=0.1):

    space_dimension = clusters_new.x[0,:].shape[0]
    print space_dimension

    for i in range (clusters_new.size):
        delta_x_Var_J = np.zeros((clusters_old.size, 3), dtype=float)
        for j in range (clusters_old.size):
            delta_x_Var_J[j,0] = (abs(clusters_new.x[i,:]-clusters_old.E_x[j,:])/clusters_new.x[i,:]).max()
            delta_x_Var_J[j,1] = (abs(clusters_new.Var_x[i,:]-clusters_old.Var_E_x[j,:])/clusters_new.Var_x[i,:]).max()
            delta_x_Var_J[j,2] = abs(clusters_new.absJ[i]-clusters_old.absJ[j])/clusters_new.absJ[i]
        i_old = np.argmin(delta_x_Var_J[:,0]) 
        print delta_x_Var_J[i_old,0] 
        if delta_x_Var_J[i_old,0] < delta_x:
            # NOTE: i_old = argmin + 1
            clusters_new.old_id[i] = i_old + 1
        else:
            clusters_new.old_id[i] = -1


