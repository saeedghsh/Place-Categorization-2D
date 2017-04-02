# Place-Categorization-2D-OGM
Semi-Supervised Place Categorization of Occupancy Grid Maps (OGM) In 2D

This repository conatins a python implementation of place categorizatio, explained in this paper - [pdf](http://www.diva-portal.org/smash/get/diva2:850141/FULLTEXT01.pdf):
```
Shahbandi, Saeed Gholami, Björn Åstrand, and Roland Philippsen. "Semi-supervised semantic labeling of adaptive cell decomposition maps in well-structured environments." Mobile Robots (ECMR), 2015 European Conference on. IEEE, 2015.
```

The feature set employed in this paper (and algorithm) is inspired by [link](http://ieeexplore.ieee.org/document/1570363/):
```
Mozos, O. Martınez, Cyrill Stachniss, and Wolfram Burgard. "Supervised learning of places from range data using adaboost." Robotics and Automation, 2005. ICRA 2005. Proceedings of the 2005 IEEE International Conference on. IEEE, 2005.
```

## Dependencies:
	* Python >=2.6
	* numpy >= 1.10.2
	* sklearn >= 0.18.0
	* opencv >= 2
	* matplotlib >= 1.4.3


## Usage:
In the [runMe.py](), set the following parameters:
```
```

Then execute the script: 
```shell
$ python runMe.py filename
```

## Notes:
* This repository contains the core for place categorization. It is not adaptive to environment types. For a better performance, one need to tweak parameters of the clustering algorithm, manually (or adaptively).
* The decomposition of the space mentioned in the paper is done by the arrangement from [this](https://github.com/saeedghsh/arrangement) repository. (which is private at the moment, send request if you need aceess.)


## License:
Distributed with a BSD license; see LICENSE.
```
Copyright (C) Saeed Gholami Shahbandi <saeed.gh.sh@gmail.com>
```
