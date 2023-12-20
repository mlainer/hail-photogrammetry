import math
import sys

#from barisal_coords import coords
from borgona_coords import coords
#from bhola_coords import coords
#from bangalore_coords import coords

from numpy import *

from qhull_2d import *
from min_bounding_rect import *

from polyplot import poly_plot

grid = []

for lon, lat in coords:

    grid.append([lon, lat])

xy_points = array(grid)

# Find convex hull
hull_points = qhull2D(xy_points)

print len(hull_points)

# Reverse order of points, to match output from other qhull implementations
hull_points = hull_points[::-1]

print 'Convex hull points: \n', hull_points, "\n"

# Find minimum area bounding rectangle
(rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)

# Verbose output of return data
print "Minimum area bounding box:"
print "Rotation angle:", rot_angle, "rad  (", rot_angle*(180/math.pi), "deg )"
print "Width:", width, " Height:", height, "  Area:", area
print "Center point: \n", center_point # numpy array
print "Corner points: \n", corner_points, "\n"  # numpy array

# Draw a nice graph with the new shape
poly_plot(array([corner_points]))