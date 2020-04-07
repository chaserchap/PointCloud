import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal

test_pipe = {
        "pipeline": [{"type": "filters.pmf"},
                     {"type": "filters.hag"},
                     {"type": "filters.eigenvalues",
                      "knn": 16},
                     {"type": "filters.normal",
                      "knn": 16}
                     ]
    }

def initialize(file):
    pipeline = {
        "pipeline": [file,
                     {"type": "filters.pmf"},
                     {"type": "filters.hag"},
                     {"type": "filters.eigenvalues",
                      "knn": 16},
                     {"type": "filters.normal",
                      "knn": 16}
                     ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline))
    if pipeline.validate():
        n_points = pipeline.execute()
        return pipeline, n_points


def zero_offset(arr):
    """Move axes to zero-based origin.

    Sets the minimum value of each axis to zero, and offsets other values accordingly.

    Expects a pipeline.array and returns a pandas dataframe object."""
    description = arr.dtype.descr
    cols = [col for col, _ in description]
    df = pd.DataFrame({col: arr[col] for col in cols})
    df['X_0'] = df['X']
    df['Y_0'] = df['Y']
    df['Z_0'] = df['Z']
    df['X'] = df['X'] - df['X_0'].min()
    df['Y'] = df['Y'] - df['Y_0'].min()
    df['Z'] = df['Z'] - df['Z_0'].min()
    return df


def visualize(df):
    n = df.shape[0]
    if n > 50000:
        df = df.loc[
            ['X', 'Y', 'Z'], np.arange(0, n, 10)]  # Note: This doesn't ensure that there are less than 50000 points...
    fig = plt.figure()
    ax = plt.axes(projection='3d', )
    ax.scatter3D(df.X, df.Y, df.Z, s=1 / 4)
    plt.show()


def set_ground(df):
    df.loc[df['HeightAboveGround'] < .2, 'Classification'] = 2
    return df


class PDALObject:

    def __init__(self, infile, outfile=None):
        self._json_pipeline = {"pipeline": []}
        self.infile = infile
        self.outfile_set = False
        self.outfile = outfile
        self.n_points = None
        self.pipeline = pdal.Pipeline(json.dumps(self.json_pipeline))
        self.executed = False
        self.arrays = None
        self.metadata = None
        self.points = None

    def add_step(self, index=-1, **kwargs):
        self.json_pipeline["pipeline"].insert(index, kwargs)
        self.pipeline = pdal.Pipeline(json.dumps(self.json_pipeline))
        if not self.pipeline.validate():
            self.json_pipeline.pop(index)
            raise Exception("Pipeline step not formatted correctly.")

    def execute(self, infile=None, outfile=None):
        if infile is not None:
            self.infile = infile
        elif self.infile is not None:
            infile = self.infile
        else:
            raise Exception("Please specify an infile.")
        if outfile is not None:
            self.outfile = outfile
        elif self.outfile is not None:
            outfile = self.outfile

        if outfile is not None:
            self.json_pipeline["pipeline"].append(outfile)

        self.json_pipeline["pipeline"].insert(0, infile)

        self.pipeline = pdal.Pipeline(json.dumps(self.json_pipeline))
        try:
            self.pipeline.validate()
        except Exception as err:
            print(err, " Please inspect PDALObject.json_pipeline")
        self.n_points = self.pipeline.execute()
        self.executed = True
        self.arrays = self.pipeline.arrays[0]
        self.metadata = json.loads(self.pipeline.metadata)
        self.json_pipeline["pipeline"].remove(infile)
        if outfile is not None:
            self.json_pipeline["pipeline"].remove(outfile)

    def reset(self):
        self.executed = False
        self.arrays = None
        self.metadata = None
        self.points = None

    @property
    def infile(self):
        return self._infile

    @infile.setter
    def infile(self, value):
        self.reset()
        self.json_pipeline = (0, value)
        self._infile = value

    @property
    def outfile(self):
        return self._outfile

    @outfile.setter
    def outfile(self, value):
        self.reset()
        self._outfile = value

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        if value is not None:
            try:
                self._points = np.vstack([value['X'], value['Y'], value['Z']]).transpose()
            except IndexError:
                self._points = np.vstack([value[0], value[1], value[2]]).transpose()
        else:
            self._points = None

    @property
    def json_pipeline(self):
        return self._json_pipeline

    @json_pipeline.setter
    def json_pipeline(self, value):
        index = -1
        try:
            index, value = value
        except:
            pass
        self._json_pipeline["pipeline"].insert(index, value)
        self.pipeline = pdal.Pipeline(json.dumps(self._json_pipeline))

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        self._pipeline = value
        self._pipeline.validate()