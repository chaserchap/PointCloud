from PointCloud import PointCloud
from pc_tools import alpha_shape, plot_polygon
from classifications import class_decode
import shapefile
import pylab as pl

def main(input, class_id, alpha=.1, output=None, show_plots=False):
    pc = PointCloud(infile=input)
    pc.classifier = 'rf_classifier.sav'
    pc.run_classifier()
    pc.group_class(class_id)
    if output is None:
        output = input.split('/')[-1].split('.')[0]
    w = shapefile.Writer('output/{0}/{0}_{1}'.format(output, class_decode[class_id]), shapeType=15)
    w.autoBalance = 1
    w.field("CLASS")
    for i in range(max(pc.class_groups)):
        z = pc.group_height(i)
        points = pc.group_geom(i)
        if show_plots==True:
            x = [p.coords.xy[0] for p in points]
            y = [p.coords.xy[1] for p in points]
        if len(points) < 5:
            continue
        concave_hull, _ = alpha_shape(points, alpha=alpha)
        counter = 0
        al = alpha
        while concave_hull.type == 'MultiPolygon' and counter < 10:
            al = al/2
            concave_hull, _ = alpha_shape(points, al)
            counter += 1
        if concave_hull.type == 'MultiPolygon':
            continue
        if len(concave_hull.bounds) > 0:
            x, y = concave_hull.exterior.coords.xy
            points = []
            for j in range(len(x)):
                points.append((x[j], y[j], z))
            w.record(class_id)
            w.polyz([points])
            if show_plots==True:
                #plot_polygon(concave_hull)
                _ = pl.plot(x, y, 'o', color='#f16824')
                pl.show()
    w.close()





