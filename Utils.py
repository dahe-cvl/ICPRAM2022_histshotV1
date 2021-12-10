import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from matplotlib import colors, colorbar
from matplotlib import pyplot as plt
import umap

########################################
### HELPERS
########################################
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def calculate_umap(name, embeddings_all_np, all_lables_np, experiment_dir="./", target_names=[]):
    
    umap = umap.UMAP(n_components=2).fit_transform(embeddings_all_np)
    
    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))
        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)
        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range	 

    tx = umap[:, 0]
    ty = umap[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    labels = np.unique(all_lables_np)
    new_cmap = rand_cmap(len(labels), type='bright', first_color_black=True, last_color_black=False, verbose=True)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # for every class, we'll add a scatter plot separately
    for i, label in enumerate(labels.tolist()):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(all_lables_np) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        #color = np.array(label, dtype=np.float) / 255
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, cmap=new_cmap, label=target_names[i], s=2)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    #plt.show()
    plt.savefig(experiment_dir + "/" + str(name) + "_umap_plt.pdf", dpi=300)

def calculate_tsne(name, embeddings_all_np, all_lables_np, experiment_dir="./", target_names=[]):
    
    all_features_np = embeddings_all_np
    tsne = TSNE(n_components=2).fit_transform(all_features_np)


    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))
        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)
        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range	 


    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    tx = np.expand_dims(tx, axis=0)
    ty = np.expand_dims(ty, axis=0)

    labels = np.unique(all_lables_np)


    new_cmap = rand_cmap(len(labels), type='bright', first_color_black=True, last_color_black=False, verbose=True)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # for every class, we'll add a scatter plot separately
    for i, label in enumerate(labels.tolist()):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(all_lables_np) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        #color = np.array(label, dtype=np.float) / 255
        
        '''
        if(label == 0):
            color = (0, 0, 0)
        elif(label == 1):
            color = (0, 1, 0)
        elif(label == 2):
            color = (0, 0, 1)
        elif(label == 3):
            color = (1, 1, 0)
        elif(label == 4):
            color = (0, 1, 1)
        elif(label == 5):
            color = (1, 0, 1)
        '''

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, cmap=new_cmap, label=target_names[i], s=2)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    #plt.show()
    plt.savefig(experiment_dir + "/" + str(name) + "_tsne_plt.pdf", dpi=300)

def plot_confusion_matrix(cm=None,
                          target_names=[],
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, path=""):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """


    fontdict = {'fontsize': 'x-large'}

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure() # figsize=(8, 6)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('acc={:0.3f}; misclass={:0.4f}'.format(accuracy, misclass), fontdict=fontdict)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize='x-large',)
        plt.yticks(tick_marks, target_names, fontsize='x-large')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize='x-large',
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize='x-large',
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.savefig(path, dpi=300)
