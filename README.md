# matrange
Visualizing matrix ranges

The MatRange class is a tool for visualizing 2x2 matrix ranges of linear operators with complex entries.  Here is an example of how to use it with the matrix [[1,0,0],[0,3,0],[0,0,5]].

```python
>>> import matrange
>>> mat = [[1,0,0],[0,3,0],[0,0,5]]
>>> X = matrange.MatRange(mat, 100, 2)
>>> X.plot_evals(X.sample, 'real', 0.5)
```

`X = matrange.MatRange(mat, 100, 2)` tells matrange to create a random sample of 100 points from the 2x2 matrix range of `mat`. The sample points are stored as a numpy array in `X.sample`.

`X.plot_evals(X.sample, 'real', 0.5)` tell matrange to plot several figures associated to the 2x2 matrix range of `mat`. From `X.sample`, it will first compute eigenvalues of the sample points.  Then it plots the convex hull of the eigenvalues, the concave hull, and the Delaunay triangulation with alpha=0.5.
