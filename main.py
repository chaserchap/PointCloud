from PointCloud import PointCloud
from src.resources.resources import *
from pc_tools import *


def main(infile, outfile, pipeline=test_pipe):
    pc = PointCloud(infile, json_pipeline=pipeline)
    pc.unassign_classification(8)

    pc.find_elevated_points()

    if pc.elevated_points.size > 0:
        powerline_info = powerlines(pc.elevated_points, ht_min=10)

        poss_lines = pc.elevated_points[powerline_info[:, 2] == 1, 3].astype(int)
        prob_lines = pc.elevated_points[powerline_info[:, 2] == 2, 3].astype(int)
        poss_poles = pc.elevated_points[powerline_info[:, 1] != -1, 3].astype(int)

        # pc.classification = (pc.eps, 7)
        pc.classification = (poss_lines, 60)
        pc.classification = (prob_lines, 60)
        pc.classification = (poss_poles, 30)
        pc.use_adj_points = False
        pc.save_las(filename=outfile)
        del pc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()
    # i = 0
    # for file in files:
    #     i += 1
    #     sys.stdout.write(file)
    #     outfile = "file{0}_final.las".format(i)
    #     main(file, outfile)
    main(args.infile, args.outfile)