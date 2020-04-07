import sys

import laspy
import numpy as np

from PDALObject import PDALObject
import pc_tools


class PointCloud:

    def __init__(self, infile=None, array=None, json_pipeline=None, use_adj_points=True, use_feet = False):
        self.infile = infile
        if infile is not None:
            self.pdal_object = PDALObject(infile)
            self.laspy_header = laspy.file.File(infile).header
        if array is None and infile is not None:
            if json_pipeline is not None:
                self.pdal_object.set_json_pipeline(json_pipeline)
            self.n_points = self.pdal_object.execute()
            self.arrays = self.pdal_object.arrays
            self.metadata = self.pdal_object.metadata
        elif array is None and infile is None:
            raise AttributeError("Please provide either an infile or an array")
        if array is not None and infile is not None:
            sys.stdout.write("Array and infile provided. Using array for points and infile for LAS header.")
            self.arrays = array
            self.n_points = array.shape[0]
        self.points = self.arrays
        self.classification = self.arrays['Classification']
        self.adj_points = self.points
        self.use_adj_points = use_adj_points
        self.use_feet = use_feet

    def save_las(self, filename):
        outFile = laspy.file.File(filename, mode='w', header=self.laspy_header)
        outFile.x = self.x
        outFile.y = self.y
        outFile.z = self.z
        outFile.raw_classification = self.classification
        outFile.close()

    def run_pdal_pipeline(self, json_pipeline):
        self.pdal_object.set_json_pipeline(json_pipeline)
        self.n_points = self.pdal_object.execute()
        self.update_points(self.pdal_object.arrays)

    def update_points(self, array):
        self.arrays = array
        self.points = array
        self.adj_points = self.points

    def find_trees_basic(self, ht_min = 2, classify=5):
        self.tree_potential = ((self.classification == 1) & (self.arrays['HeightAboveGround'] >= ht_min)
                                & (self.arrays['Eigenvalue0'] > 0.5)
                                & (self.arrays['NumberOfReturns'] - self.arrays['ReturnNumber'] >= 1))
        if classify is not None:
            self.classification = (self.tree_potential, classify)
        return self.tree_potential

    def find_buildings_basic(self, ht_min=7, classify=True):
        self.roof_mask = ((self.classification == 1)
                          & (self.arrays['HeightAboveGround'] > ht_min)
                          & (self.arrays['Eigenvalue0'] <= .02)
                          & (self.arrays['NumberOfReturns'] == self.arrays['ReturnNumber']))
        if classify:
            self.classification = (self.roof_mask, 6)
        return self.roof_mask

    def unassign_classification(self, class_to_unassign=None):
        if class_to_unassign is None:
            self.classification[:] = 1
        else:
            self.classification[:] = class_to_unassign

    def find_elevated_points(self, ht_min = 40):
        if 'HeightAboveGround' in self.arrays.dtype.names:
            self.eps = self.arrays['HeightAboveGround'] > ht_min
        else:
            self.eps = pc_tools.ground_points_grid_filter(self.points, z_height=ht_min)

    @property
    def elevated_points(self):
        if self.eps is not None:
            return self.points[self.eps]
        else:
            return None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        if value is not None:
            point_index = np.arange(0, len(value))
            try:
                self._points = np.vstack([value['X'], value['Y'], value['Z'], point_index]).transpose()
            except IndexError:
                self._points = np.vstack([value[:, 0], value[:, 1], value[:, 2], point_index]).transpose()
        else:
            self._points = None

    @property
    def x(self):
        if self.use_adj_points:
            return self._adj_points[:, 0]
        else:
            return self.points[:, 0]

    @property
    def y(self):
        if self.use_adj_points:
            return self._adj_points[:, 1]
        else:
            return self.points[:, 1]

    @property
    def z(self):
        if self.use_adj_points:
            return self._adj_points[:, 2]
        else:
            return self._points[:, 2]

    @property
    def classification(self):
        return self._classification

    @classification.setter
    def classification(self, value):
        try:
            array, value = value
            self._classification[array] = value
        except ValueError:
            self._classification = value

    def update_class_by_array(self, array, value):
        self._classification[array] = value

    @property
    def adj_points(self):
        if self._adj_points is not None:
            return self._adj_points
        elif self.points is not None:
            self.adj_points = self.points
            return self._adj_points
        else:
            return None

    @adj_points.setter
    def adj_points(self, value):
        if value is not None:
            x = value[:, 0].min()
            y = value[:, 1].min()
            z = value[:, 2].min()
            self._adj_points = np.array([value[:, 0] - x, value[:, 1] - y, value[:, 2] - z]).transpose()
        else:
            self._adj_points = None
