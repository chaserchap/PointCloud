from pathlib import Path
import shutil

import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

import sys

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def ground_points_grid_filter(dataset, z_height=None, z_factor=10, n=None, x_step=10, y_step=10):
    """Determine ground points using a grid.

    Prevents identification of large hills/mountains as VOs by identifying height ranges at a more granular level."""
    dataset_z_filtered = dataset[[0]]

    if z_height is not None:
        z_filter = z_height
    elif z_factor is not None and z_height is None:
        z_filter = (dataset[:, 2].max() - dataset[:, 2].min()) / z_factor
    else:
        raise ValueError("Please provide a value for either z_height or z_factor, but not both.")

    point_index = np.arange(0, len(dataset.shape[0]))
    dataset = np.insert(dataset, 3, point_index, axis=1)

    if n is not None:
        sys.stdout.write("Using n to determine xstep and ystep.\n")
        xstep = (dataset[:, 0].max() - dataset[:, 0].min()) / n
        ystep = (dataset[:, 1].max() - dataset[:, 1].min()) / n
    else:
        sys.stdout.write("Using x_step and y_step values, inferring from data to establish m2 boxes...\n")
        xstep = (dataset[:, 0].max() - dataset[:, 0].min()) / round((dataset[:, 0].max() - dataset[:, 0].min())/x_step)
        ystep = (dataset[:, 1].max() - dataset[:, 1].min()) / round((dataset[:, 1].max() - dataset[:, 1].min())/y_step)

    sys.stdout.write("Filtering points {0} meters above ground level in {1} m^2 subsets of original data.\n".format(
        z_filter, xstep*ystep))

    for x in frange(dataset[:, 0].min(), dataset[:, 0].max(), xstep):
        for y in frange(dataset[:, 1].min(), dataset[:, 1].max(), ystep):
            dataset_filtered = dataset[(dataset[:, 0] > x)
                                      & (dataset[:, 0] < x + xstep)
                                      & (dataset[:, 1] > y)
                                      & (dataset[:, 1] < y + ystep)]

            if dataset_filtered.shape[0] > 0:
                dataset_filtered = dataset_filtered[dataset_filtered[:, 2] > (dataset_filtered[:, 2].min() + z_filter)]

                if dataset_filtered.shape[0] > 0:
                    dataset_z_filtered = np.concatenate((dataset_z_filtered, dataset_filtered))

    sys.stdout.write("Found {0} points exceeding z-threshold.".format(dataset_z_filtered.shape[0]))

    point_index[:] = 0
    point_index[dataset[:, 3].astype(int)] = 1

    return point_index.astype(bool)

def dbscan_cluster(dataset, eps=None, min_samples=5, normalize = False):
    if normalize:
        # Original example normalized data. Not sure I agree with doing so. Clustering shouldn't be too dependent on
        # all values being within similar ranges.
        dataset = preprocessing.normalize(dataset)
    if eps is None:
        eps = dataset.median() - 1e-8 # Subtract to place eps below median (even if barely).
        # Best would likely be to determine eps with respect to the dataset, following some kind of distance graph.
        # Might be able to determine a good setting based on metaparameters. Such as eps from lidar point spacing.
        #
        # Could possibly run db scan twice, once to pick up all points with small eps and large k. This would find power
        # poles and lines (potentially.) Run again with a smaller k and ideally the lines would get dropped...
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)
    labels = clustering.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    sys.stdout.write("Found {0} clusters in the data. With {1} noise points.\n".format(n_clusters_, n_noise_))
    return clustering

def visualize(dataset, cluster, labels = None):

    if labels is not None and dataset.shape[0] != len(labels):
        raise ValueError("Number of labels not equal to number of data points.")

    if labels is None:
        labels = cluster.labels_

    core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)

    core_samples_mask[cluster.core_sample_indices_] = True

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    fig = plt.figure(figsize=[100, 50])

    ax = fig.add_subplot(111, projection='3d')
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xyz = dataset[class_member_mask & core_samples_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, marker=".")
    plt.title('Estimated number of cluster: %d' % n_clusters_)
    plt.show()

def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def check_pole(data, width=20):
    """Determine if the data cluster is a pole.
    This presumes a pole is less than width feet/meter diameter. Note: data should already be identified as an
    elevated point."""
    xy = data[:, 0:2]
    centroid = centroidnp(xy)
    radii = np.sqrt(np.sum(np.square(xy-centroid), axis=1))
    if radii.max() > width:
        return False
    return True

def powerlines(filtered_dataset):
    """Run dbscan twice to identify powerlines vs powerpoles."""
    line_cluster = dbscan_cluster(filtered_dataset[:,0:2], eps=50, min_samples=4) # eps and min_samples may vary...
    pole_cluster = dbscan_cluster(filtered_dataset[:,0:2], eps=10, min_samples=10)

    pole_labels = pole_cluster.labels_
    pole_unique_labels = set(pole_labels)
    poles = np.full(len(pole_labels), -1).astype(int)
    for label in pole_unique_labels:
        group = pole_labels == label
        if check_pole(filtered_dataset[group]):
            poles[group] = label

    all_labels = line_cluster.labels_
    all_unique = set(all_labels)
    powerlines = np.full(len(all_labels), -1)

    for label in all_unique:
        group = all_labels == label
        poles_in_group = poles[group]
        if len(set(poles_in_group)) > 2.:
            powerlines[group] = 2.0
        elif len(set(poles_in_group)) > 1.:
            powerlines[group] = 1.0

    return np.vstack((all_labels, poles, powerlines)).transpose()

def process(file, create_filtered_las = True, z_feet=30, z_factor=None, n=None, x_step=15, y_step=15, destination=None,
            debug=False, del_temp = True):
    inFile = laspy.file.File(file)
    dataset = np.vstack([inFile.x, inFile.y, inFile.z]).transpose()
    if debug:
        del_temp = False

    sys.stdout.write("Processing {0} points. \nX max:{1} min:{2} \nY max:{3} min:{4}\nZ max:{5} min:{6}\n".format(
        dataset.shape[0], dataset[:, 0].max(), dataset[:, 0].min(), dataset[:, 1].max(), dataset[:, 1].min(),
        dataset[:, 2].max(), dataset[:, 2].min()))

    dataset = ground_points_grid_filter(dataset, z_height=z_feet, z_factor=z_factor, n=n, x_step=x_step, y_step=y_step)

    poles = powerlines(dataset)

    clustering = dbscan_cluster(dataset, eps=50, min_samples=4)

    if create_filtered_las:
        # outfile = file.split('/')[-1][:-4] + '_eps.las'
        # outFile = laspy.file.File(outfile, mode='w', header=inFile.header)
        # outFile.x = dataset[:, 0]
        # outFile.y = dataset[:, 1]
        # outFile.z = dataset[:, 2]
        # outFile.close()
        create_las_file_from_xyz(file, inFile.header, dataset, suffix='_eps', destination=destination,
                                 del_temp=del_temp)

    return dataset, clustering, poles

def new_file(filename, suffix, extension=None):
    path = Path(filename)
    if extension is None:
        extension = str(path.suffix)
    name = str(path.stem) + suffix + extension
    return path.parent.joinpath(name)

def create_las_file_from_xyz(filename, header, dataset, suffix='_example', destination = None, del_temp = True):
    outfile_full = new_file(filename, suffix)
    outfile_temp = Path.cwd().joinpath("tmp/" + str(outfile_full.name))
    outFile = laspy.file.File(outfile_temp, mode='w', header=header)
    outFile.x = dataset[:, 0]
    outFile.y = dataset[:, 1]
    outFile.z = dataset[:, 2]
    outFile.close()
    if destination is None:
        final_destination = outfile_full
    else:
        final_destination = Path(destination).parent.joinpath(outfile_temp.name)
    if del_temp:
        shutil.move(outfile_temp, final_destination)
    else:
        shutil.copy(outfile_temp, final_destination)
    sys.stdout.write("Created file at {0}.\n".format(final_destination))


def create_results(file, z_feet=None, z_factor=10, n=None, x_step=15, y_step=15, powerlines=1, destination=None,
                   debug=False):
    dataset, cluster, poles = process(file, z_feet=z_feet, z_factor=z_factor, n=n, x_step=x_step, y_step=y_step,
                                      debug=debug)
    inFile = laspy.file.File(file)
    pole_file = file.split('/')[-1][:-4] + '_poles.las'
    line_file = file.split('/')[-1][:-4] + '_lines.las'

    if debug:
        del_temp=False

    pole_points = poles[:, 1]
    if len(np.unique(pole_points)) > 1:
        # poleFile = laspy.file.File(pole_file, mode='w', header=inFile.header)
        # poleFile.x = dataset[pole_points > 0][:, 0]
        # poleFile.y = dataset[pole_points > 0][:, 1]
        # poleFile.z = dataset[pole_points > 0][:, 2]
        # poleFile.close()
        data = dataset[pole_points > 0]
        create_las_file_from_xyz(file, inFile.header, data, suffix='_poles', destination=destination, del_temp=del_temp)

    else:
        sys.stdout.write("No poles identified. Pole file not created.\n")

    powerlines -= 1

    line_points = poles[:, 2] > powerlines
    if sum(line_points) > 0:
        # lineFile = laspy.file.File(line_file, mode='w', header=inFile.header)
        # lineFile.x = dataset[line_points][:, 0]
        # lineFile.y = dataset[line_points][:, 1]
        # lineFile.z = dataset[line_points][:, 2]
        # lineFile.close()
        data = dataset[line_points]
        create_las_file_from_xyz(file, inFile.header, data, suffix='_lines', destination=destination, del_temp=del_temp)

    else:
        sys.stdout.write("No lines identified. Line file not created.\n")

    # non_ground = line_points > powerlines or pole_points > 0

    # groundfile = file.split('/')[-1][:-4] + '_ground.las'
    # groundFile = laspy.file.File(ground_file, mode='w', header=inFile.header)
    # groundFile.x = dataset[ground_file][:, 0]
    # groundFile.y = dataset[ground_file][:, ]

if __name__ == "__main__":
    destination = "/mnt/RECON/AeroResults/SanDiego"
    for file in files:
        create_results(file, z_feet=50, destination=destination, debug=True)