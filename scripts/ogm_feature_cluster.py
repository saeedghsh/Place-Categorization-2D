import sys
import cv2
import numpy as np
import sklearn.cluster

new_paths = [
    u'/home/saesha/Dropbox/myGits/arrangement/',
    # u'/home/saesha/Dropbox/myGits/Python-CPD/'
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import arrangement.arrangement as arr
reload(arr)
import arrangement.utils as utls
reload(utls)


# import matplotlib.pyplot as plt


def deploy_arrangement(file_name, config={'multi_processing':4,
                                          'end_point':False,
                                          'timing':False}):
    ''' '''
    try:
        data = arr.utls.load_data_from_yaml( file_name[:-4]+'.yaml' )
        traits = data['traits']
        boundary = data['boundary']
        boundary[0] -= 20
        boundary[1] -= 20
        boundary[2] += 20
        boundary[3] += 20
        
        ### trimming traits
        traits = utls.unbound_traits(traits)
        traits = arr.utls.bound_traits(traits, boundary)
        
        ### deploying arrangement
        arrange = arr.Arrangement(traits, config)
    except:
        print ('yaml file not found') 
        arrange = None

    return arrange

def main(file_name, n_categories=4):
    ### loading image, features, and opencells 
    features =   np.load( file_name[:-4]+'_{:s}.npy'.format('features') )
    open_cells = np.load( file_name[:-4]+'_{:s}.npy'.format('open_cells') )
    image = np.flipud( cv2.imread( file_name, cv2.IMREAD_GRAYSCALE) )


    # selecting only those cells inside the arrangement
    arrangement = deploy_arrangement(file_name)
    if arrangement is not None:
        in_arrangement = [False]*open_cells.shape[0]
        open_cells_col_row = np.stack( (open_cells[:,1],
                                        open_cells[:,0]),
                                       axis=1)
        for sf in arrangement._get_independent_superfaces():
            in_arrangement = np.logical_or( in_arrangement,
                                            sf.path.contains_points(open_cells_col_row))
            
        open_cells = open_cells[in_arrangement,:]
        features = features[in_arrangement,:]


    X = features
    ### rejectin NaNs
    X = np.where ( np.isnan(X), np.zeros(X.shape) , X) 
    
    ### Normalizing 
    for i in range(X.shape[1]):
        X[:,i] /= X[:,i].mean()

    ### clustering
    kmean = sklearn.cluster.KMeans(n_clusters=n_categories,
                                   precompute_distances=False,
                                   n_init=20, max_iter=500).fit(X)
    labels = kmean.labels_

    ### generating and storing labeling image
    label_image = np.ones(image.shape) *-1 # initializing every pixel as outlier
    label_image[open_cells[:,0],open_cells[:,1]] = labels

    # # plotting 
    # fig, axes = plt.subplots(1,1, figsize=(20,12))
    # axes.imshow(label_image, origin='lower')
    # plt.show()

    # saving
    np.save( file_name[:-4]+'_labels_km{:s}.npy'.format(str(n_categories)),
             label_image)


if __name__ == '__main__':
    '''
    python ogm_feature_cluster.py --file_name map_sample/E5_layout.png --n_categories 2
    python ogm_feature_cluster.py --file_name /home/saesha/Documents/tango/HIH_01_full/20170131135829.png --n_categories 2
    '''
    args = sys.argv
    file_name = None

    # fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument   
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = listiterator.next()
            if item[:2] == '--':
                exec(item[2:] + ' = listiterator.next()')
        except:
            break

    # if file name is not provided, set to default
    if file_name is None:
        raise (NameError('no file name is found'))

    if 'n_categories' in locals():
        n_categories = int(n_categories)
    else:
        n_categories = 4

    main(file_name, n_categories)
