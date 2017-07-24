Place-Categorization-2D-OGM
===========================
Semi-Supervised Place Categorization of Occupancy Grid Maps (OGM) In 2D

This repository conatins a python implementation of place categorization method, explained in this paper ([link](http://ieeexplore.ieee.org/document/7324207/), [pdf](http://www.diva-portal.org/smash/get/diva2:850141/FULLTEXT01.pdf)):
- Shahbandi, Saeed Gholami, Björn Åstrand, and Roland Philippsen. "Semi-supervised semantic labeling of adaptive cell decomposition maps in well-structured environments." Mobile Robots (ECMR), 2015 European Conference on. IEEE, 2015.  

The feature set employed in this work is inspired by [link](http://ieeexplore.ieee.org/document/1570363/):
- Oscar Martinez Mozos "Semantic labeling of places with mobile robots" 2010, Springer Berlin Heidelberg.
<!-- - Mozos, O. Martınez, Cyrill Stachniss, and Wolfram Burgard. "Supervised learning of places from range data using adaboost." Robotics and Automation, 2005. ICRA 2005. Proceedings of the 2005 IEEE International Conference on. IEEE, 2005. -->

Image below shows a simple demo of place categorization, with Kmean clustering where number of categories are set to two.
For more details and more examples see the abovementioned paper.
![cover](https://github.com/saeedghsh/Place-Categorization-2D/blob/master/docs/HIH.png)

Dependencies
-------------
To install dependencies:
```shell
git clone https://github.com/saeedghsh/Place-Categorization-2D.git
cd Place-Categorization-2D
pip install -r requirements.txt % opencv must be installed separately
```

Notes
-----
- This repository contains the core method for place categorization.
  It is not adaptive to environment types, and paramters must be set according to input maps (e.g. resolution).
  For a better performance, one need to tweak parameters of the clustering algorithm, manually (or adaptively).
- The decomposition of the 2D plane from the abovementioned paper ("Semi-supervised ...") is carried out by the [arrangement](https://github.com/saeedghsh/arrangement) package.


License
-------
Distributed with a GNU license; see LICENSE.
```
Copyright (C) Saeed Gholami Shahbandi <saeed.gh.sh@gmail.com>
```

<!-- Laundry List -->
<!-- ------------ -->
<!-- - [x] `feature_testing.py` -->
<!-- - numbers of gaps seems to be THE MOST RELEVANT AND DOMINANT features. the best result I get comes from that. -->
<!-- - The first 8 features also works, but very noisy and sensitive -->
<!-- - PCA stuff (x4) kinda works too, with different result. -->
<!-- - PCA stuff (x3) of resampled also works with defferent result. -->
<!--   resampled PCA stuff (x4) didn't work at all at first, -->
<!--   Turned out the zero value of PCA in resampled version makes huge values in ratio feature, which in turn screws the clustering. -->
<!-- - [ ] alternative to multiprocessing, compromised vectorization: -->
<!--   check the memory of the machine ([psutil](https://pypi.python.org/pypi/psutil)). -->
<!--   find the appropriate batch size for each method according to the memory available. -->
<!--   and then deploy each method in a batch, with determined size, in vectorized manner. -->

<!-- Paramters -->
<!-- --------- -->
<!-- Set the following parameters: -->
<!-- ```python -->
<!-- # raycasting -->
<!-- mpp = 0.02, # meter per pixel -->
<!-- range_meter   = 8, # meter -->
<!-- length_range  = 400, #range_meter_ / mpp_ -->
<!-- length_steps  = 400, #int(length_range_) -->
<!-- theta_range   = 2*np.pi, -->
<!-- theta_res     = 1/1, # step/degree -->
<!-- occupancy_thr = 210, -->

<!-- # feature extraction -->
<!-- gapThreshold  = [1.0] # in meter -->

<!-- # clustering -->
<!-- ``` -->

<!-- Scripts -->
<!-- ------- -->
<!-- - `raycast_demo.py`   -->
<!--   execute the following to see how raycasting works: -->
<!--   ```shell -->
<!--   python raycast_demo.py --image_name 'filename.ext' -->
<!--   ``` -->

<!-- - `raycast_map.py`   -->
<!--   Results are stored in a `.npy` file. -->
<!--   To load the results somewhere else: -->
<!--   ```python -->
<!--   raycasts = np.atleast_1d( np.load('filename') )[0] -->
<!--   ``` -->
<!--   Where `raycasts` is a a dictionary with `['config', 'open_cells', 'theta_vecs', 'range_vecs']` for keys. -->
<!--   `raycasts['config']` contains the raycasting configurations. -->
<!--   `raycasts['open_cells']` is a 2d array of open cell coordinates (`x,y`) in image frame. -->
<!--   For an `open_cell_idx`, the ray cast from that point is given by -->
<!-- 	  - `t = raycasts['theta_vecs']` -->
<!-- 	  - `r = raycasts['range_vecs'][open_cell_idx,:]` -->

<!-- - `raycast_map_batch.py`   -->
<!--   This was an attempt to vectorize the raycasting process. -->
<!--   I tried to raycast all points in open space at once. -->
<!--   It works, but it requires huge memory and hence becomes super slow as the open space grows. -->
<!--   I two examples, for ~1000 cells it tooks about 3 seconds and for ~2800 cells it took about a minute! -->

<!-- - `cluster_features.py`   -->
<!-- As the final stage of place categorization, this script loads features from file and performs clustering on them. -->

<!-- Vectorization with numpy and multi-processing -->
<!-- --------------------------------------------- -->
<!-- Both raycasting and feature extraction methods are implemented with numpy vectorization. -->
<!-- However, these methods demand a good deal of memory. -->
<!-- Thus, it is maed possible to execute these methods in a sequantial manner, with multiproccessing. -->
<!-- Raycasting is worse than feature extraction in terms of memory foot-print. -->
<!-- On laptop with 8Gb of RAM, I had to deployed the multiproccessing version of raycasting. -->
<!-- But the feature extraction did work with vectorization. -->
<!-- I suspect for big maps, even the feature extraction would run out memory and needs a batch version. -->
