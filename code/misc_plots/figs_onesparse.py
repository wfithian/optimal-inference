import itertools
import numpy as np
from scipy.spatial import (voronoi_plot_2d, Voronoi, 
                           ConvexHull, convex_hull_plot_2d)
from scipy.spatial._plotutils import _adjust_bounds
from matplotlib import pyplot
from selection.affine import constraints, sample_from_constraints
import os
from scipy.stats import norm

figl, figw = 9.5, 9.5
# data point
Y = [-.45, .85]

def hull_without_points(hull, ax=None):
    """
    modified from scipy.spatial.convex_hull_plot_2d
    """
    if ax is None:
        ax = pyplot.gcf().gca()
        
    if hull.points.shape[1] != 2:
        raise ValueError("Convex hull is not 2-D")

    for simplex in hull.simplices:
        ax.plot(hull.points[simplex,0], hull.points[simplex,1], 'k-')

    _adjust_bounds(ax, hull.points)

    return ax.figure

def angle(x):
    """
    recover angle from a 2 vector
    """
    theta = np.arccos(x[0] / np.linalg.norm(x))
    if x[1] < 0:
        theta = 2 * np.pi - theta
    return theta

def just_hull(W, fill=True, fill_args={}, label=None, ax=None, ptlabel=None):
    
    """
    Draw the hull without points
    """
    hull = ConvexHull(W)
    f = hull_without_points(hull, ax=ax)
    a = f.gca()
    a.set_xticks([])
    a.set_yticks([])

    A, b, pairs, angles, perimeter, ai = extract_constraints(hull)
    if fill:
        perimeter_vertices = np.array([v for _, v in perimeter])

        pyplot.scatter(perimeter_vertices[:,0],
                       perimeter_vertices[:,1], c='gray', s=100, label=ptlabel)
        pyplot.fill(perimeter_vertices[:,0], perimeter_vertices[:,1],
                    label=label, **fill_args)
    a.scatter(0,0, marker='+', c='k', s=50)
    return f, A, b, pairs, angles, perimeter, ai 
    
def extract_constraints(hull):
    """
    given a convex hull, extract

    (A,b) such that

    $$hull = \{x: Ax+b \geq 0 \}$$

    also, return rays of the normal cone associated to each vertex as `pairs`

    """
    A = []
    b = []
    pairs = []
    angles = []

    perimeter = []

    angle_intersection = []

    for simplex1, simplex2 in itertools.combinations(hull.simplices, 2):
        intersect = set(simplex1).intersection(simplex2)

        for p in simplex1:
            perimeter.append((angle(hull.points[p]), list(hull.points[p])))

        for p in simplex2:
            perimeter.append((angle(hull.points[p]), list(hull.points[p])))

        if intersect:
            v1, v2 = hull.points[[simplex1[0], simplex1[1]]]
            diff = v1-v2
            normal1 = np.array([diff[1],-diff[0]])

            # find a point not in the simplex
            i = 0
            while True:
                s = hull.points[i]
                if i not in simplex1:
                    break
                i += 1    
            if np.dot(normal1, s-hull.points[simplex1[0]]) > 0:
                normal1 = -normal1
                
            v1, v2 = hull.points[[simplex2[0], simplex2[1]]]
            diff = v1-v2
            normal2 = np.array([diff[1],-diff[0]])
            
            # find a point not in the simplex
            i = 0
            while True:
                s = hull.points[i]
                if i not in simplex2:
                    break
                i += 1
                
            if np.dot(normal2, s-hull.points[simplex2[0]]) > 0:
                normal2 = -normal2
                
            dual_basis = np.vstack([normal1, normal2])
            angles.extend([angle(normal1), angle(normal2)])
            angle_intersection.append([angle(normal1), angle(normal2), intersect])
            pairs.append((hull.points[list(intersect)[0]], dual_basis))

    for simplex in hull.simplices:
        v1, v2 = hull.points[[simplex[0], simplex[1]]]
        diff = v1-v2
        normal = np.array([diff[1],-diff[0]])
        offset = -np.dot(normal, v1)
        scale = np.linalg.norm(normal)
        if offset < 0:
            scale *= -1
            normal /= scale
            offset /= scale
        A.append(normal)
        b.append(offset)

    # crude rounding
    angles = np.array(angles)
    angles *= 50000
    angles = np.unique(angles.astype(np.int))
    angles = angles / 50000.

    return np.array(A), np.array(b), pairs, angles, sorted(perimeter), angle_intersection

symmetric = False
np.random.seed(10)
W = np.array([(np.cos(a), np.sin(a)) for a in sorted([0.4,3.3,4.0, 5.1])])
if symmetric: 
    W = np.vstack([W,-W])
hull = ConvexHull(W)
f, A, b, pairs, angles, perimeter, ai = just_hull(W, 
                                                  fill_args={'facecolor':'gray', 'alpha':0.2}, 
                                                  label=r'$K$')

def cone_rays(angles, ai, hull, which=None, ax=None, fill_args={},
              plot=True):
   """

   Plot the given Voronoi diagram in 2-D based on a set of directions

   Parameters
   ----------
   vor : scipy.spatial.Voronoi instance
       Diagram to plot
   ax : matplotlib.axes.Axes instance, optional
       Axes to plot on

   Returns
   -------
   fig : matplotlib.figure.Figure instance
       Figure for the plot

   See Also
   --------
   Voronoi

   Notes
   -----
   Requires Matplotlib.

   """
   angles = np.sort(angles)
   points = np.array([np.cos(angles), np.sin(angles)]).T
   
   vor = Voronoi(points)
   
   if vor.points.shape[1] != 2:
       raise ValueError("Voronoi diagram is not 2-D")

   if ax is None:
       ax = pyplot.gca()

   rays = np.array([(np.cos(_angle), np.sin(_angle)) for _angle in angles])
   for i in range(rays.shape[0]):
       rays[i] /= np.linalg.norm(rays[i])
   rays *= 100
   
   if plot:
       for ray in rays:
           ax.plot([0,ray[0]],[0,ray[1]], 'k--')
   
   if which is not None:
       if which < rays.shape[0]-1:
           active_rays = [rays[which], rays[which+1]]    
       else:
           active_rays = [rays[0], rays[-1]]
       poly = np.vstack([active_rays[0], np.zeros(2), active_rays[1], 100*(active_rays[0]+active_rays[1])])
       dual_rays = np.linalg.pinv(np.array(active_rays))
       angle1 = angle(dual_rays[:,0])
       angle2 = angle(dual_rays[:,1])

   else:
       poly = None
       active_rays = None
       dual_rays = None

   _adjust_bounds(ax, vor.points)
   
   ax.set_xticks([])
   ax.set_yticks([])
   return ax, poly, dual_rays, np.array(active_rays)

def all_dual_rays(W):

   f, A, b, pairs, angles, perimeter, ai = just_hull(W)
    
   angles = np.sort(angles)
   points = np.array([np.cos(angles), np.sin(angles)]).T
   
   vor = Voronoi(points)
   if vor.points.shape[1] != 2:
       raise ValueError("Voronoi diagram is not 2-D")

   rays = np.array([(np.cos(_angle), np.sin(_angle)) for _angle in angles])
   for i in range(rays.shape[0]):
       rays[i] /= np.linalg.norm(rays[i])
   rays  *= 100

   active_rays = []
   dual_rays = []
   for i in range(rays.shape[0]):
       if i < rays.shape[0] - 1:
           active_rays.append(np.array([rays[i], rays[i+1]]))
       else:
           active_rays.append(np.array([rays[0], rays[-1]]))
       dual_rays.append(np.linalg.pinv(active_rays[-1]))
   return angles, active_rays, dual_rays, rays

angles, A, D = all_dual_rays(W)[:3]
signs = np.array([np.sign(np.dot(d.T, Y)) for d in D])
region = np.argmax(signs.sum(1))

def cone_with_slice(angles, ai, hull, which, fill_args={}, ax=None, label=None,
                    suffix='', 
                    Y=None):

    ax, poly, constraint, rays = cone_rays(angles, ai, hull, which, ax=ax, fill_args=fill_args)
    eta_idx = np.argmax(np.dot(hull.points, Y))
    eta = 40 * hull.points[eta_idx]

    representation = constraints(-constraint.T, np.zeros(2))

    if Y is None:
        Y = sample_from_constraints(representation)

    ax.fill(poly[:,0], poly[:,1], label=r'$A_{(M,H_0)}$', **fill_args)
    if symmetric:
        ax.fill(-poly[:,0], -poly[:,1], **fill_args)

    legend_args = {'scatterpoints':1, 'fontsize':30, 'loc':'lower left'}
    ax.legend(**legend_args)
    ax.figure.savefig('fig_onesparse1.png', dpi=300)

    ax.scatter(Y[0], Y[1], c='k', s=150, label=label)

    Vp, _, Vm = representation.bounds(eta, Y)[:3]

    Yperp = Y - (np.dot(eta, Y) / 
                 np.linalg.norm(eta)**2 * eta)

    if Vm == np.inf:
        Vm = 10000

    width_points = np.array([(Yperp + Vp*eta /  
                              np.linalg.norm(eta)**2),
                             (Yperp + Vm*eta /  
                              np.linalg.norm(eta)**2)])

    ax.plot(width_points[:,0], width_points[:,1], '-', c='k', linewidth=4)
    legend_args = {'scatterpoints':1, 'fontsize':30, 'loc':'lower left'}
    ax.legend(**legend_args)
    ax.figure.savefig('fig_onesparse2.png', dpi=300)

    return ax, poly, constraint, rays

f = pyplot.figure(figsize=(figl,figw))
ax = f.gca()
ax = cone_with_slice(angles,
                     ai, 
                     hull,
                     region,
                     ax=ax,
                     label=r'$y$',
                     Y=Y,
                     fill_args=\
                  {'facecolor':'gray', 'alpha':0.2})[0]
restriction = ax.figure
restriction.savefig('cone_with_slice.png')
