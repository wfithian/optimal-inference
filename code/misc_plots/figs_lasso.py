import itertools
import numpy as np
from scipy.spatial import (voronoi_plot_2d, Voronoi, 
                           ConvexHull, convex_hull_plot_2d)
from scipy.spatial._plotutils import _adjust_bounds
from matplotlib import pyplot
import matplotlib.colors
from selection.affine import constraints, sample_from_constraints
import os
from scipy.stats import norm
from selection.lasso import lasso

figl, figw = 9.5, 9.5
# data point
Y = [1.3,2]

yellow = matplotlib.colors.colorConverter.to_rgb("#f4e918")

def hull_without_points(hull, ax=None):
    """
    modified from scipy.spatial.convex_hull_plot_2d
    """
    if ax is None:
        ax = pyplot.gcf().gca()
        
    if hull.points.shape[1] != 2:
        raise ValueError("Convex hull is not 2-D")

    for simplex in hull.simplices:
        ax.plot(hull.points[simplex,0], hull.points[simplex,1], 'k--')

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

def just_hull(W, fill_args={}, label=None, ax=None, ptlabel=None,
              vertices=True, fill=True):
    
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

        if fill:
            pyplot.fill(perimeter_vertices[:,0], perimeter_vertices[:,1],
                        label=label, **fill_args)
    if vertices:
        pyplot.scatter(perimeter_vertices[:,0],
                       perimeter_vertices[:,1], c=yellow, s=100, label=ptlabel)

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

symmetric = True
np.random.seed(10)
W = np.array([(np.cos(a), np.sin(a)) for a in (np.arange(0, np.pi, np.pi/3) + np.random.sample(3) * 0.6)])
if symmetric: 
    W = np.vstack([W,-W])
hull = ConvexHull(W)

def cone_rays(angles, ai, hull, which=None, ax=None, fill_args={},
              plot=True):
   """
   draw the cone rays
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
   
   if plot:
       for ray in rays:
           ax.plot([0,100*ray[0]],[0,100*ray[1]], 'k--')
   
   if which is not None:
       if which < rays.shape[0]-1:
           active_rays = [100*rays[which], 100*rays[which+1]]    
       else:
           active_rays = [100*rays[0], 100*rays[-1]]
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

   active_rays = []
   dual_rays = []
   for i in range(rays.shape[0]):
       if i < rays.shape[0] - 1:
           active_rays.append(np.array([rays[i], rays[i+1]]))
       else:
           active_rays.append(np.array([rays[0], rays[-1]]))
       dual_rays.append(np.linalg.pinv(active_rays[-1]))
   return angles, active_rays, dual_rays, rays, A, b

angles, _, Dual, rays, A, b = all_dual_rays(W)
signs = np.array([np.sign(np.dot(d.T, Y)) for d in Dual])

# this part finds hyperplanes that can be used to form design 
# X for LASSO.

# there is no guarantee that the [0,2,3] will work for other (Y,W)
# it was done by inspection here

A /= b[:,None]
X = -A[[0,2,3]].T

region = np.argmax(signs.sum(1))

def hull_with_rays(W, fill=True, fill_args={}, label=None, ax=None,
                    Y=None, vertices=True, which=region):
    f, A, b, pairs, angles, perimeter, ai = just_hull(W,
                                                      fill=fill,
                                                      label=label,
                                                      ax=ax,
                                                      fill_args=fill_args,
                                                      vertices=vertices)
    

    ax, poly, constraint, rays = cone_rays(angles, ai, hull, which, ax=ax, plot=False, fill_args=fill_args)    

    if Y is not None:
        L = lasso(Y, X, lam=1.)
        L.fit(min_its=200, tol=1.e-14)
        representation = L.constraints
        eta = L._XEinv[1]

    vtx_idx = np.argmax(np.dot(hull.points, Y))
    vtx = hull.points[vtx_idx]

    for i in range(len(pairs)):
        v, D = pairs[i]
        ax.plot([v[0],v[0]+10000*D[0,0]],[v[1],v[1]+10000*D[0,1]], 'k--')
        ax.plot([v[0],v[0]+10000*D[1,0]],[v[1],v[1]+10000*D[1,1]], 'k--')
    
    ax.set_xlim(3*np.array(ax.get_xlim()))
    ax.set_ylim(3*np.array(ax.get_ylim()))
    legend_args = {'scatterpoints':1, 'fontsize':25, 'loc':'lower left'}
    ax.legend(**legend_args)

    for i in range(3):
        ax.arrow(0,0,X[0,i]/3.,X[1,i]/3., linewidth=3, head_width=0.02, fc='k')

    Vp, V, Vm = representation.bounds(eta, Y)[:3]

    Yperp = Y - (np.dot(eta, Y) / 
                 np.linalg.norm(eta)**2 * eta)

    if Vm == np.inf:
        Vm = 10000

    slice_points = np.array([(Yperp + Vp*eta /  
                              np.linalg.norm(eta)**2),
                             (Yperp + Vm*eta /  
                              np.linalg.norm(eta)**2)])


    ax.legend(**legend_args)

    ax.text(0.01,0.48, r'$X_3$', fontsize=25)
    ax.text(0.39,0.15, r'$X_1$', fontsize=25)
    ax.text(-0.57,-0.30, r'$X_2$', fontsize=25)

    f.savefig('fig_lasso3.pdf')
    f.savefig('fig_lasso3.png')

    ax.fill(poly[:,0] + vtx[0], poly[:,1] + vtx[1], label=r'$\{1,3\}$ selected', **fill_args)
    ax.fill(-poly[:,0] - vtx[0], -poly[:,1] - vtx[1], **fill_args)

    f.savefig('fig_lasso0.pdf')
    f.savefig('fig_lasso0.png')
    ax.text(1.4,2.1, r'$Y$', fontsize=25)
    ax.add_patch(pyplot.Circle(Y, radius=.08, facecolor='k'))
    f.savefig('fig_lasso1.pdf')
    f.savefig('fig_lasso1.png')

    ax.plot(slice_points[:,0] - 0*V, slice_points[:,1] - 0*V, '-', c='k', linewidth=4)
    ax.plot([Y[0]]*2, [Y[1]]*2, c='k')
    f.savefig('fig_lasso2.pdf')
    f.savefig('fig_lasso2.png')

    return f, A, b, pairs, angles, perimeter 

f = pyplot.figure(figsize=(figl,figw))
f.clf()
ax = f.gca()
polytope_with_cones = hull_with_rays(W, 
                                     fill_args=( 
                                     {'facecolor':yellow, 'alpha':0.8}), 
                                     ax=f.gca(),
                                     label=r'$K$',
                                     fill=False,
                                     vertices=False,
                                     Y=Y)[0]


