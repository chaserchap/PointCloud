from pathlib import Path
import shutil

import laspy
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from descartes import PolygonPatch
from scipy.spatial import Delaunay
from shapely.ops import cascaded_union, polygonize
import shapely.geometry as geometry
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

import sys
import math
import warnings

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def ground_points_grid_filter(dataset, z_height=None, z_factor=10, n=None, x_step=10, y_step=10):
    """Determine ground points using a grid.

    Prevents identification of large hills/mountains as VOs by identifying height ranges at a more granular level."""


    if z_height is not None:
        z_filter = z_height
    elif z_factor is not None and z_height is None:
        z_filter = (dataset[:, 2].max() - dataset[:, 2].min()) / z_factor
    else:
        raise ValueError("Please provide a value for either z_height or z_factor, but not both.")

    point_index = np.arange(0, dataset.shape[0])
    dataset = np.insert(dataset, 3, point_index, axis=1)
    dataset_z_filtered = dataset[[0]]

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
    point_index[dataset_z_filtered[:, 3].astype(int)] = 1

    return point_index.astype(bool)

def dbscan_cluster(dataset, eps=None, min_samples=5, normalize = False):
    if normalize:
        # Original example normalized data. Not sure I agree with doing so. Clustering shouldn't be too dependent on
        # all values being within similar ranges.
        dataset = preprocessing.normalize(dataset)
    if eps is None:
        eps = 10 # Using the median as below resulted in too large an epsilon. Need to find a better estimate.
        # eps = np.median(dataset[:, 0:3]) - 1e-8 # Subtract to place eps below median (even if barely).
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

def visualize(dataset, cluster=None, labels = None):

    if labels is not None and dataset.shape[0] != len(labels):
        raise ValueError("Number of labels not equal to number of data points.")
    if labels is None and cluster is not None:
        labels = cluster.labels_
    elif labels is None and cluster is None:
        labels = np.zeros(dataset.shape[0])

    core_samples_mask = np.zeros_like(labels, dtype=bool)

    if cluster is not None:
        core_samples_mask[cluster.core_sample_indices_] = True
    else:
        core_samples_mask[:] = True

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
    if filtered_dataset.size == 0:
        return False
    line_cluster = dbscan_cluster(filtered_dataset[:,0:2], eps=50, min_samples=4) # eps and min_samples may vary...
    pole_cluster = dbscan_cluster(filtered_dataset[:,0:3], eps=20, min_samples=15)

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

def geometric_median(X, numIter = 200):
    """
    Compute the geometric median of a point sample.
    The geometric median coordinates will be expressed in the Spatial Image reference system (not in real world metrics).
    We use the Weiszfeld's algorithm (http://en.wikipedia.org/wiki/Geometric_median)

    :Parameters:
     - `X` (list|np.array) - voxels coordinate (3xN matrix)
     - `numIter` (int) - limit the length of the search for global optimum

    :Return:
     - np.array((x,y,z)): geometric median of the coordinates;
    """
    # -- Initialising 'median' to the centroid
    y = np.mean(X, 1)
    # -- If the init point is in the set of points, we shift it:

    if X.shape[1] == 3:
        three_d = True
        while (y[0] in X[:, 0]) and (y[1] in X[:, 1]) and (y[2] in X[:, 2]):
            y+=0.1
    elif X.shape[0] == 2:
        three_d = False
        while (y[0] in X[:, 0]) and (y[1] in X[:, 1]):
            y += 0.1

    convergence = False  # boolean testing the convergence toward a global optimum
    dist = []  # list recording the distance evolution

    # -- Minimizing the sum of the squares of the distances between each points in 'X' and the median.
    i = 0
    while (not convergence) and (i < numIter):
        num_x, num_y, num_z = 0.0, 0.0, 0.0
        denum = 0.0
        m = X.shape[1]
        d = 0
        for j in range(0,m):
            if three_d:
                div = math.sqrt((X[j, 0]-y[0])**2 + (X[j, 1]-y[1])**2 + (X[j, 2]-y[2])**2)
                num_z += X[j, 2] / div
            else:
                div = math.sqrt((X[j, 0]-y[0])**2 + (X[j, 1]-y[1])**2)
            num_x += X[j, 0] / div
            num_y += X[j, 1] / div
            denum += 1./div
            d += div**2  # distance (to the median) to miminize
        dist.append(d)  # update of the distance evolution

        if denum == 0.:
            warnings.warn("Couldn't compute a geometric median, please check your data!")
            return [0, 0, 0]

        if three_d:
            y = [num_x/denum, num_y/denum, num_z/denum]  # update to the new value of the median
        else:
            y = [num_x/denum, num_y/denum]
        if i > 3:
            convergence = (abs(dist[i]-dist[i-2]) < 0.1)  # we test the convergence over three steps for stability
            #~ print abs(dist[i]-dist[i-2]), convergence
        i += 1
    if i == numIter:
        raise ValueError("The Weiszfeld's algoritm did not converged after"+str(numIter)+"iterations !!!!!!!!!" )
    # -- When convergence or iterations limit is reached we assume that we found the median.

    return np.array(y)

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])
    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter
        # print circum_r
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def plot_polygon(polygon):
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return fig

if __name__ == "__main__":
    destination = "/mnt/RECON/AeroResults/SanDiego"
    for file in files:
        create_results(file, z_feet=50, destination=destination, debug=True)