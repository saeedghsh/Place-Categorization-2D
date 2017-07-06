Place-Categorization-2D-OGM
===========================
Semi-Supervised Place Categorization of Occupancy Grid Maps (OGM) In 2D

This repository conatins a python implementation of place categorizatio, explained in this paper ([link](http://ieeexplore.ieee.org/document/7324207/), [pdf](http://www.diva-portal.org/smash/get/diva2:850141/FULLTEXT01.pdf)):
- Shahbandi, Saeed Gholami, Björn Åstrand, and Roland Philippsen. "Semi-supervised semantic labeling of adaptive cell decomposition maps in well-structured environments." Mobile Robots (ECMR), 2015 European Conference on. IEEE, 2015.

The feature set employed in this work is inspired by [link](http://ieeexplore.ieee.org/document/1570363/):
- Oscar Martinez Mozos "Semantic labeling of places with mobile robots" 2010, Springer Berlin Heidelberg.
<!-- - Mozos, O. Martınez, Cyrill Stachniss, and Wolfram Burgard. "Supervised learning of places from range data using adaboost." Robotics and Automation, 2005. ICRA 2005. Proceedings of the 2005 IEEE International Conference on. IEEE, 2005. -->

![cover](https://github.com/saeedghsh/Place-Categorization-2D/blob/master/docs/E5.png)

Dependencies
-------------
To install dependencies:
```shell
git clone https://github.com/saeedghsh/Place-Categorization-2D.git
cd Place-Categorization-2D
pip install -r requirements.txt
```

Notes
-----
- This repository contains the core functionality for place categorization.
  It is not adaptive to environment types, and paramters must be set according to input maps (e.g. resolution).
  For a better performance, one need to tweak parameters of the clustering algorithm, manually (or adaptively).
- The decomposition of the 2D plane mentioned in the paper ("Semi-supervised ...") is carried out by the [arrangement](https://github.com/saeedghsh/arrangement) package.


Paramters
---------
Set the following parameters:
```python
# raycasting

# feature extraction

# clustering
```
Then execute on of the script mentioned in the next part.


Scripts
-------
- `raycast_demo.py`  
  execute the following to see how raycasting works:
  ```shell
  python raycast_demo.py --image_name 'filename.ext'
  ```

- `raycast_map.py`  
  Results are stored in a `.npy` file.
  To load the results somewhere else:
  ```python
  raycasts = np.atleast_1d( np.load('filename') )[0]
  ```
  Where `raycasts` is a a dictionary with `['config', 'open_cells', 'theta_vecs', 'range_vecs']` for keys.
  `raycasts['config']` contains the raycasting configurations.
  `raycasts['open_cells']` is a 2d array of open cell coordinates (`x,y`) in image frame.
  For an `open_cell_idx`, the ray cast from that point is given by
	  - `t = raycasts['theta_vecs']`
	  - `r = raycasts['range_vecs'][open_cell_idx,:]`

- `raycast_map_batch.py`  
  This was an attempt to vectorize the raycasting process.
  I tried to raycast all points in open space at once.
  It works, but it requires huge memory and hence becomes super slow as the open space grows.
  I two examples, for ~1000 cells it tooks about 3 seconds and for ~2800 cells it took about a minute!

- `cluster_features.py`  
As the final stage of place categorization, this script loads features from file and performs clustering on them.

License
-------
Distributed with a GNU license; see LICENSE.
```
Copyright (C) Saeed Gholami Shahbandi <saeed.gh.sh@gmail.com>
```

<!-- Laundry List -->
<!-- ------------ -->
<!-- - [ ] todo -->
