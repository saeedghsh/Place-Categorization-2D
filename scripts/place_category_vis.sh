#!/bin/bash -ue

path_list=(
    '/home/saesha/Documents/tango/kpt4a_f/'
    '/home/saesha/Documents/tango/kpt4a_kb/'
    '/home/saesha/Documents/tango/kpt4a_kl/'
    '/home/saesha/Documents/tango/kpt4a_lb/'
    
    '/home/saesha/Documents/tango/HIH_01_full/'
    
    '/home/saesha/Documents/tango/E5_1/'
    '/home/saesha/Documents/tango/E5_2/'
    '/home/saesha/Documents/tango/E5_3/'
    '/home/saesha/Documents/tango/E5_4/'
    '/home/saesha/Documents/tango/E5_5/'
    '/home/saesha/Documents/tango/E5_6/'
    '/home/saesha/Documents/tango/E5_7/'
    '/home/saesha/Documents/tango/E5_8/'
    '/home/saesha/Documents/tango/E5_9/'
    '/home/saesha/Documents/tango/E5_10/'

    '/home/saesha/Documents/tango/F5_1/'
    '/home/saesha/Documents/tango/F5_2/'
    '/home/saesha/Documents/tango/F5_3/'
    '/home/saesha/Documents/tango/F5_4/'
    '/home/saesha/Documents/tango/F5_5/'
    '/home/saesha/Documents/tango/F5_6/'


    # layouts
    '/home/saesha/Dropbox/myGits/sample_data/HH/E5/'
    '/home/saesha/Dropbox/myGits/sample_data/HH/F5/'
    '/home/saesha/Dropbox/myGits/sample_data/HH/HIH/'
    '/home/saesha/Dropbox/myGits/sample_data/sweet_home/'
)

for path in "${path_list[@]}"; do

    echo "current path: $path"
    file_list=(`ls $path*.npy`)
    python ogm_cluster_vis.py ${file_list[1]} ${file_list[2]} ${file_list[3]} ${file_list[4]}
    
    # for file in "${file_list[@]}"; do
    # 	# <STRING> == <PATTERN>
    # 	# <STRING> is checked against the pattern <PATTERN> - TRUE on a match
    # 	# But note, quoting the pattern forces a literal comparison.
    # 	if [[ "$file" == *_labels_km* ]]; then
    # 	    python ogm_cluster_vis.py $file
	    
    # 	fi
    # done

done
