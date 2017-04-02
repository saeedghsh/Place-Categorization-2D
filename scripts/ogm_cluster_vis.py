import sys
import numpy as np
import matplotlib.pyplot as plt

def main_signle(file_name):
    fig, ax = plt.subplots()
    label_image = np.load(file_name)
    ax.imshow(label_image, interpolation='nearest', origin='lower')
    ax.set_title(file_name)
    plt.tight_layout()
    plt.show()


def main_quadruple(file_names):    
    fig, axes = plt.subplots(2,2, figsize=(20,12))
    
    label_image = np.load(file_names[0])        
    axes[0,0].imshow(label_image, interpolation='nearest', origin='lower')
    title = file_names[0].split('.')[0].split('/')[-1][-3:]
    axes[0,0].set_title(title)
    axes[0,0].axis('off')

    label_image = np.load(file_names[1])        
    axes[0,1].imshow(label_image, interpolation='nearest', origin='lower')
    title = file_names[1].split('.')[0].split('/')[-1][-3:]
    axes[0,1].set_title(title)
    axes[0,1].axis('off')

    label_image = np.load(file_names[2])        
    axes[1,0].imshow(label_image, interpolation='nearest', origin='lower')
    title = file_names[2].split('.')[0].split('/')[-1][-3:]
    axes[1,0].set_title(title)
    axes[1,0].axis('off')

    label_image = np.load(file_names[3])        
    axes[1,1].imshow(label_image, interpolation='nearest', origin='lower')
    title = file_names[3].split('.')[0].split('/')[-1][-3:]
    axes[1,1].set_title(title)
    axes[1,1].axis('off')

    fig_name = file_names[0].split('.')[0].split('/')[-1][:-4]+'.png'
    
    plt.savefig('examples/'+fig_name, bbox_inches='tight')

    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    args = sys.argv
    
    if len(args)==2:
        file_name = args[1]
        main_signle (file_name)

    elif len(args)==5:
        file_names = args[1:]
        main_quadruple (file_names)

    else:
        print ('\n\t*** NO FILE IS SPECIFIED ***')
