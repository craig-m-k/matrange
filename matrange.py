"""
Generates and analyzes 2x2 spatial matricial ranges.

Parts of alpha_shape, and plot_polygon, are from Kevin Dwyer:
http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
"""

import numpy as np
from descartes import PolygonPatch
from scipy import linalg
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import math
from matplotlib.collections import LineCollection


def plot_polygon(polygon):
    fig = plt.figure()
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


class MatRange(object):
    """
    A collection of methods for analyzing spatial matricial ranges of
    matrices.

    generate_sample: Generates numPoints s x s matrices of the spatial
    matricial range of self.matrix. All s x s compressions can be
    obtained by conjugating self.matrix by a unitary, then compressing
    to the top left corner. This routine creates numPoints uniformly 
    distributed unitaries with which to create the sample.
    
    plot_evals:  we can't directly plot a complex 2x2 matrix, so we
    take the real part and find the eigenvalues and plot those.
    """
    def __init__(self, matrix, num_points, dim, flatten):
        self.matrix = np.matrix(matrix)
        self.size = self.matrix.shape
        self.sample = self._generate_sample(num_points, dim, flatten)

    def _generate_sample(self, num_points, dim, flatten=False):
        """
        Sample num_points from the dim x dim spatial matricial range. If 
        flatten is True, the sample point arrays are flattened.
        """
       
        isom = np.matrix(np.zeros((self.size[1],dim))) #isometry
        for i in range(dim):
            isom[i,i] = 1
        points = []
        for i in range(num_points):
            # Method 1 for random unitary

            # Generate random complex matrix
            r_matrix = (np.matrix(np.random.randn(self.size[0], self.size[1]))
                + 1j*np.matrix(np.random.randn(self.size[0], self.size[1])))

            # From r_matrix form a self-adjoint matrix
            h_matrix = (r_matrix+r_matrix.H)/2

            # From h_matrix form a unitary
            u_matrix = np.matrix(linalg.expm(1j*h_matrix)) # unitary

            """
            Another method for random unitary:
            z = (np.random.randn(self.size[0],self.size[1]) +
                 1j*np.random.randn(self.size[0],self.size[1]))/math.sqrt(2.0)
            q,r = linalg.qr(z)
            d = np.diagonal(r)
            ph = d/np.absolute(d)
            U = np.matrix(np.multiply(q,ph,q))
            """
            
            # Compress self.matrix
            compression = np.array(isom.H*u_matrix.H*self.matrix*u_matrix*isom)

            if flatten is False:
                points.append(compression)
            else:
                r_part = np.real(compression).flatten()
                i_part = np.imag(compression).flatten()
                flat_point = np.concatenate((r_part, i_part), axis=0)
                points.append(flat_point)
        points = np.array(points)
        return points

    def _alpha_shape(self, coords, alpha):
        # Compute alpha shape
        
        def add_edge(edges, edge_points, coords, i, j):
            # Add line between i-th and j-th point
            if (i, j) in edges or (j, i) in edges:
                return
            edges.add((i,j))
            edge_points.append(coords[[i,j]])

        tri = Delaunay(coords)
        totalarea = 0
        edges = set()
        edge_points = []

        # loop over triangles
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]
            
            # squared lengths of sides of triangle
            a2 = (pa[0]-pb[0])**2 + (pa[1]-pb[1])**2
            b2 = (pb[0]-pc[0])**2 + (pb[1]-pc[1])**2
            c2 = (pc[0]-pa[0])**2 + (pc[1]-pa[1])**2
            
            # squared area with Heron's formula
            A2 = (1./16)*(4*(a2*b2+a2*c2+b2*c2) - (a2+b2+c2)**2)
            
            circum_r2 = a2*b2*c2/(4.0*A2)
            # radius filter
            if circum_r2 < 1.0/(alpha**2):
                totalarea += math.sqrt(A2)
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)
        m = geometry.MultiLineString(edge_points)
        triangles = list(polygonize(m))
        return cascaded_union(triangles), edge_points, totalarea
        
    def plot_evals(self, points, etype, alpha):
        # Plot evals of the sample point matrices.
        
        # Exit if sample matrices aren't 2x2
        if points[0].shape != (2,2):
            raise ValueError("Shape: {0}.\
                Expected:(2,2).".format(points[0].shape))

        # List of evals to plot
        delaunay_points = []
        for p in points:
            if etype == 'cplx':
                q = -1j*p
            else:
                q = p
            e = np.real(np.linalg.eig(q)[0])
            
            # Matricial ranges are symmetric
            delaunay_points.append([e[0],e[1]])
            delaunay_points.append([e[1],e[0]])
        delaunay_points = np.array(delaunay_points)

        # Plot convex hull
        hull = ConvexHull(delaunay_points)
        for simplex in hull.simplices:
            plt.plot(delaunay_points[simplex, 0],\
                     delaunay_points[simplex, 1], 'k-')
        plt.title('Convex hull')
        plt.show()

        # Plot Delaunay triangulation
        concave_hull, edge_points, area = \
            self._alpha_shape(delaunay_points, alpha=alpha)
        lines = LineCollection(edge_points)
        plt.figure()
        title1 = """{1} Delaunay triangulation with 
        {2} points, alpha={0}""".format(alpha,etype,2*len(points))
        plt.title(title1)
        plt.gca().add_collection(lines)
        plt.plot(delaunay_points[:,0], delaunay_points[:,1],
                    'o', hold=1, color='#f16824')

        # Plot concave hull
        _ = plot_polygon(concave_hull)
        _ = plt.plot(delaunay_points[:,0],delaunay_points[:,1],\
                     'o', color='#f16824')
        _ = plt.title('Concave hull, alpha = {0}'.format(alpha))
            
        plt.show()

