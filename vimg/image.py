# coding=utf-8
""" An image viewer for the command line. """
from __future__ import division
import os
import sys
import math
import cv2
import numpy as np
import random
import multiprocessing as mp

PY2 = sys.version_info < (3,0)

##
## Use multiprocessing?
##
FLAG_MP = True

##
## FONT_ASPECT is the height-to-width ratio of a character slot in the terminal.
##
FONT_ASPECT = 30./14
##
## CHANNEL_VALUES are the values that each RGB channel can take in the set of xterm-256 colors.
##
CHANNEL_VALUES = np.expand_dims((0x00, 0x5f, 0x87, 0xaf, 0xd7, 0xff),axis=-1)
##
## Create grayscale lookup dictionaries
##
GRAYSCALE_VALUES = np.array([0x08, 0x12, 0x1c, 0x26, 0x30, 0x3a, 0x44, 0x4e, 0x58, 0x62, 0x6c, 0x76, 0x80, 0x8a, 0x94, 0x9e, 0xa8, 0xb2, 0xbc, 0xc6, 0xd0, 0xda, 0xe4, 0xee])
GRAYSCALE_INT = np.array([232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255])
GRAYSCALE_LOOKUP = dict(zip(GRAYSCALE_VALUES,GRAYSCALE_INT))
GRAYSCALE_REVERSE_LOOKUP = dict(zip(GRAYSCALE_INT,GRAYSCALE_VALUES))
##
## Characters used to mix two colors at a given ratio
##
ASCII_CHARS_SET = [
    (u' ',),            # 0
    (u'\u2581', u'\u258F', u'\u2595', u'\u2594'),      # 1/8
    (u'\u2582', u'\u258E', u'\u2596', u'\u2597', u'\u2598', u'\u259D'),      # 1/4
    # (u'\u2591',), # 1/4
    (u'\u2583', u'\u258D'),      # 3/8
    (u'\u2584', u'\u258C', u'\u2590', u'\u2580', u'\u259A', u'\u259E'),      # 1/2
    # (u'\u2592',), # 1/2
    (u'\u2585', u'\u258B'),      # 5/8
    (u'\u2586', u'\u258A', u'\u2599', u'\u259B', u'\u259C', u'\u259F'),      # 3/4
    # (u'\u2593',), # 3/4
    (u'\u2587', u'\u2589'),      # 7/8
    (u'\u2588',)       # 1
]
##
## Characters to represent pixels
##
CELLSIZE = (8,4)        # rows,columns
CHAR_TEMPLATES = [
    # (character, mask with weights)
    (u'▁', np.transpose([[ 4, 4, 4, 3, 3, 1, 1,-1]])),      # Lower one eighth block
    (u'▂', np.transpose([[ 4, 4, 3, 3, 1, 1,-1,-1]])),      # Lower one quarter block
    (u'▃', np.transpose([[ 4, 3, 3, 1, 1,-1,-1,-3]])),      # Lower three eighths block
    (u'▄', np.transpose([[ 3, 3, 1, 1,-1,-1,-3,-3]])),      # Lower half block
    (u'▅', np.transpose([[ 3, 1, 1,-1,-1,-3,-3,-4]])),      # Lower five eighths block
    (u'▆', np.transpose([[ 1, 1,-1,-1,-3,-3,-4,-4]])),      # Lower three quarters block
    (u'▇', np.transpose([[ 1,-1,-1,-3,-3,-4,-4,-4]])),      # Lower seven eighths block
    (u'■', np.transpose([[ 1, 1,-1,-2,-2,-1, 1, 1]])),      # Middle half block
    (u'▉', np.array([[-4,-4,-4,-3,-3,-1,-1, 1]])),          # Left seven eighths block
    (u'▊', np.array([[-4,-4,-3,-3,-1,-1, 1, 1]])),          # Left three quarters block
    (u'▋', np.array([[-4,-3,-3,-1,-1, 1, 1, 3]])),          # Left five eighths block
    (u'▌', np.array([[-3,-3,-1,-1, 1, 1, 3, 3]])),          # Left half block
    (u'▍', np.array([[-3,-1,-1, 1, 1, 3, 3, 4]])),          # Left three eighths block
    (u'▎', np.array([[-1,-1, 1, 1, 3, 3, 4, 4]])),          # Left one quarter block
    (u'▏', np.array([[-1, 1, 1, 3, 3, 4, 4, 4]])),          # Left one eighth block
    (u'▖', np.array([[ 1, 2],[-1, 1]])),                    # Quadrant lower left
    (u'▗', np.array([[ 2, 1],[ 1,-1]])),                    # Quadrant lower right
    (u'▘', np.array([[-1, 1],[ 1, 2]])),                    # Quadrant upper left
    (u'▝', np.array([[ 1,-1],[ 2, 1]])),                    # Quadrant upper right
    (u'▞', np.array([[ 1,-1],[-1, 1]])),                    # Quadrant upper right and lower left
]

def groupby(values):
    diff = np.concatenate(([1],np.diff(values)))
    idx = np.concatenate((np.where(diff)[0],[len(values)]))
    index = np.empty(len(idx)-1,dtype='i8,u2')
    index['f0']=values[idx[:-1]]
    index['f1']=np.diff(idx)
    return index

def inflate_array(array, shape):
    """ Inflate array to given shape using constant (order 0) interpolation.

    Parameters
    ----------
    array : numpy.array
        The numpy array to be inflated.
    shape : tuple
        The requested shape. The components of `shape` must be integer divisible by the components
        of the array shape.

    Returns
    -------
    numpy.array
        A numpy array of shape `shape`.
    """
    if 1 in array.shape:
        ##
        ## Reduce to smallest possible representation
        ##
        grouped = groupby(array.flatten())
        counts = np.unique(grouped['f1'])
        if len(counts) == 1:
            if array.shape[0] == 1:
                new_shape = (1, int(array.shape[1] / counts[0]))
            else:
                new_shape = (int(array.shape[0] / counts[0]), 1)
            array = grouped['f0'].reshape(new_shape)
    zoomy, zoomx = np.array(shape) / array.shape
    zoomy = int(zoomy)
    zoomx = int(zoomx)
    return np.repeat(np.repeat(array,zoomy,axis=0),zoomx,axis=1)

##
## Create masks with CELLSIZE shape
##
CHARS = []
for (c,m) in CHAR_TEMPLATES:
    ##
    ## Skip characters where the resolution doesn't match the cellsize:
    ##
    if ((m < 0).sum(axis=0) / float(m.shape[0]) * CELLSIZE[0] % 1).any() or \
       ((m < 0).sum(axis=1) / float(m.shape[1]) * CELLSIZE[1] % 1).any():
        continue
    ratio = (m==0).sum() / m.size
    w = inflate_array(m, CELLSIZE)
    mask = (w < 0)
    weights1 = np.where(mask,w,0)
    weights0 = np.where(~mask,w,0)
    weights1 = weights1 / weights1.sum()
    weights0 = weights0 / weights0.sum()
    weights1 = np.expand_dims(weights1, axis=-1)
    weights0 = np.expand_dims(weights0, axis=-1)
    CHARS.append((c,ratio,mask,weights1,weights0))

##
## Turn into array
##
WEIGHTS_1 = np.stack([w1 for _,_,m,w1,w0 in CHARS],axis=-1)     # shape: (Y,X,1,N)
WEIGHTS_0 = np.stack([w0 for _,_,m,w1,w0 in CHARS],axis=-1)     # shape: (Y,X,1,N)

##
## COLOR CONVERSION HELPERS
##

def rgb_bracket(rgb):
    """ Find the closest two xterm-256 approximations that can be mixed to yield the given RGB value.

    Parameters
    ----------
    rgb : numpy.array
        An RGB array, i.e. three integers between 0 and 255

    Returns
    -------
    tuple(int,int,float)
        A tuple of the integer representations of two xterm-256 compatible colors and the mixing ratio
        that yields the best approximation to the given color.
    """
    rgb1 = rgb_closest(rgb, asint=False)
    rgb2 = rgb_closest(2*rgb - rgb1, asint=False)
    d1 = colordiff(rgb,rgb1)
    d2 = colordiff(rgb,rgb2)
    if d1==0 and d2==0:
        ratio = 1.0
    else:
        ratio = d2 / (d1+d2)
    return rgb_lookup(rgb1), rgb_lookup(rgb2), ratio

def rgb_closest(rgb,asint=True):
    """ Find the closest available xterm-256 approximation to the RGB color.

    Parameters
    ----------
    rgb : numpy.array
        An RGB array, i.e. three integers between 0 and 255
    asint : bool, optional
        Whether to return the integer representation instead of an RGB tuple (default: True)

    Returns
    -------
    int
        The integer representation of the closest xterm-256 color. If asint is False, returns an RGB
        tuple instead.
    """
    diff = abs(CHANNEL_VALUES - rgb)
    xterm_color = CHANNEL_VALUES[diff.argmin(axis=0)].flatten()
    ##
    ## Consider grayscale
    ##
    mindiff = colordiff(rgb,xterm_color)
    rgb_mean = rgb.sum()/rgb.size
    graydiff = abs(GRAYSCALE_VALUES - rgb_mean)
    gray = GRAYSCALE_VALUES[graydiff.argmin()]
    rgb_gray = (gray,gray,gray)
    if colordiff(rgb,rgb_gray) < mindiff:
        xterm_color = rgb_gray
    if asint:
        return rgb_lookup(xterm_color)
    else:
        return xterm_color

def rgb_reverse_lookup(index):
    """ Find the RGB value of the xterm-256 color associated with the given integer.

    Parameters
    ----------
    index : int
        The integer representation of an xterm-256 color.

    Returns
    -------
    numpy.array
        A numpy.array object of length 3 with the RGB values.
    """
    if index in GRAYSCALE_REVERSE_LOOKUP:
        gray = GRAYSCALE_REVERSE_LOOKUP[index]
        return (gray,gray,gray)
    index -= 16
    remainder1 = index % 36
    remainder2 = remainder1 % 6
    pos = np.array([
        (index - remainder1) / 36,
        (remainder1 - remainder2) / 6,
        remainder2
    ], dtype=np.int)
    return CHANNEL_VALUES[pos].flatten()

def rgb_lookup(rgb):
    """ Return the integer representation of an xterm-256 compatible RGB color.

    Parameters
    ----------
    rgb : numpy.array
        numpy.array of RGB channels, each between 0 and 255

    Returns
    -------
    int
        The xterm-256 integer representing the color, or False if the color doesn't exist.
    """
    if rgb[0]==rgb[1] and rgb[0]==rgb[2] and rgb[0] in GRAYSCALE_LOOKUP:
        return GRAYSCALE_LOOKUP[rgb[0]]
    pos = (CHANNEL_VALUES == rgb).argmax(axis=0)
    return 16 + 36 * pos[0] + 6 * pos[1] + pos[2]

def rgb2color(rgb):
    """ Like rgb_bracket, but instead of the mixing ratio returns a character with appropriate
    filling ratio.

    Parameters
    ----------
    rgb : numpy.array
        numpy.array of RGB channels, each between 0 and 255

    Returns
    -------
    tuple(int,int,char)
        A tuple of the integer representations of two xterm-256 colors for background and foreground
        and the character to represent the mixing ratio.
    """
    bg_col, fg_col, ratio = rgb_bracket(rgb)
    ##
    ## ratio is the mixing ratio (always >= 0.5)
    ## --> The smaller the ratio, the larger the foreground character needs to be.
    ##
    return (bg_col, fg_col, ratio2char(ratio))

def ratio2char(ratio):
    """ Returns a character that fills a character cell to approximately 1 minus the given ratio.

    Parameters
    ----------
    ratio : float
        The ratio of the cell to be filled by the character, between 0 and 1.

    Returns
    -------
    char
        A character that fills a character cell to approximately 1 minus the given ratio.
    """
    ##
    ## The first char should correspond to 0% foreground coverage.
    ## The last char should correspond to 100% foreground coverage.
    ##
    index = int( np.around( (1.0-ratio) * (len(ASCII_CHARS_SET) - 1) ) )
    charset = ASCII_CHARS_SET[index]
    char = random.choice(charset)
    return char

def gray2char(gray):
    """ Like ratio2char, but expects an integer value between 0 and 255. Maps a character representing
    the grayscale value of the cell.

    Parameters
    ----------
    gray : int
        A grayscale value between 0 and 255.

    Returns
    -------
    char
        A character that represents the grayscale value.
    """
    if type(gray) is np.int:
        gray = gray/256.0
    index = min(len(ASCII_CHARS_SET) - 1, int(gray * len(ASCII_CHARS_SET)))
    charset = ASCII_CHARS_SET[index]
    char = random.choice(charset)
    return char

def colordiff(rgb1,rgb2):
    """ Computes the Euclidean difference between two RGB color tuples.

    Parameters
    ----------
    rgb1 : numpy.array
        First RGB color
    rgb2 : numpy.array
        Second RGB color

    Returns
    -------
    float
        The Euclidean distance between the two colors.
    """
    r1,g1,b1=rgb1
    r2,g2,b2=rgb2
    return (float(r1)-float(r2))**2 + (float(g1)-float(g2))**2 + (float(b1)-float(b2))**2

def pixels2cell(pixels):
    """ Convert a pixel array to the best possible representation by a character, background
    and foregrund color.

    Parameters
    ----------
    pixels : numpy.array
        numpy.array representing RGB pixels. Must be reshapable into CELLSIZE+(3,)

    Returns
    -------
    tuple(int,int,char)
        The optimal background color, foreground color and character to represent the two pixels in
        a single character cell.
    """
    if not pixels.shape == CELLSIZE + (3,1,):
        pixels = pixels.reshape( CELLSIZE + (3,1,) )
    ##
    ## Compute the contrast between
    ##
    char = u' '
    bg_color_rgb = None
    fg_color_rgb = None
    bg_color_approx = None
    fg_color_approx = None
    max_sep = None
    max_colordiff = None

    ##
    ## Choose mask with best inter-pixel contrast
    ##
    # (Y,X) is the cellsize
    # N is the number of available characters for the chosen cellsize
    p_1 = pixels * WEIGHTS_1        # shape: (Y,X,3,N)
    p_0 = pixels * WEIGHTS_0        # shape: (Y,X,3,N)
    # compute means
    rgb_1 = p_1.sum(axis=(0,1))     # shape: (3,N)
    rgb_0 = p_0.sum(axis=(0,1))     # shape: (3,N)
    # compute variances (exclude 0 weights)
    var_1 = np.where(WEIGHTS_1 != 0, pixels - rgb_1, 0)**2  # shape: (Y,X,3,N)
    var_0 = np.where(WEIGHTS_0 != 0, pixels - rgb_0, 0)**2  # shape: (Y,X,3,N)
    # reduce variance dimension
    var_1 = var_1.sum(axis=(0,1,2))                         # shape: (N,)
    var_0 = var_0.sum(axis=(0,1,2))                         # shape: (N,)
    var_within = var_1 + var_0                              # shape: (N,)
    # deal with 0 variance
    zero_var = (var_within == 0)
    if zero_var.any():
        # Return first entry where the variance is zero
        max_index = zero_var.argmax()
        max_colordiff = colordiff(rgb_1[:,max_index],rgb_0[:,max_index])
    else:
        # Compute contrast measure per mask
        contrast = ((rgb_1 - rgb_0)**2).sum(axis=0)         # shape: (N,)
        sep = contrast / var_within                         # shape: (N,)
        max_index = sep.argmax()                            # scalar
        max_sep = sep[max_index]                            # scalar
        max_colordiff = contrast[max_index]                 # scalar

    # Results
    char = CHARS[max_index][0]
    fg_color_rgb = rgb_1[:,max_index]
    bg_color_rgb = rgb_0[:,max_index]

    bg_color_approx = rgb_closest(bg_color_rgb,asint=False)
    fg_color_approx = rgb_closest(fg_color_rgb,asint=False)

    if max_colordiff > 40 or \
      ( colordiff(bg_color_rgb,bg_color_approx) < 10 and colordiff(fg_color_rgb,fg_color_approx) < 10 ):
        ##
        ## If the two cell part colors sufficiently accurate OR
        ## very different from each other, display them as different pixels.
        ##
        fg_color = rgb_lookup(fg_color_approx)
        bg_color = rgb_lookup(bg_color_approx)

    else:
        ##
        ## Else, try to improve color accuracy by mixing colors.
        ##
        rgb = pixels.mean(axis=(0,1,3))
        bg_color, fg_color, ratio = rgb_bracket(rgb)
        char = ratio2char(ratio)

    return bg_color, fg_color, char


def best_representation(values):
    """ Given two RGB colors as an RGBRGB 6-tuple, representing two vertically adjacent pixels,
    finds the best possible colored representation of these pixels in a single character cell.

    This method seeks the optimal trade-off between spatial resolution and color accuracy.

    Parameters
    ----------
    values : numpy.array
        The RGB values of two vertically stacked pixels as a single numpy array of length 6.

    Returns
    -------
    tuple(int,int,char)
        The optimal background color, foreground color and character to represent the two pixels in
        a single character cell.
    """
    upper = values[:3]
    lower = values[3:]

    upper_closest = rgb_closest(upper,asint=False)
    lower_closest = rgb_closest(lower,asint=False)

    if ( colordiff(upper,upper_closest) < 10 and colordiff(lower,lower_closest) < 10 ) \
        or colordiff(upper,lower) > 50:
        ##
        ## If upper_pixel and lower_pixel are sufficiently accurate or
        ## very different from each other, display them as different pixels.
        ##
        bg_color = rgb_lookup(upper_closest)
        fg_color = rgb_lookup(lower_closest)
        char = u'\u2584'
    else:
        ##
        ## Else, try to improve color accuracy by mixing colors.
        ##
        rgb = 0.5*upper + 0.5*lower
        bg_color, fg_color, ratio = rgb_bracket(rgb)
        char = ratio2char(ratio)

    return bg_color, fg_color, char


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """ Like numpy.apply_along_axis(), but takes advantage of multiple cores.

    Taken from https://stackoverflow.com/a/45555516
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, mp.cpu_count())]

    pool = mp.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

def unpacking_apply_along_axis(params):
    """ Like numpy.apply_along_axis(), but with arguments in a tuple instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().

    Taken from https://stackoverflow.com/a/45555516
    """
    (func1d, axis, arr, args, kwargs) = params
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


#########################################################################
# IMAGE #################################################################
#########################################################################

class Image:
    """ An Image class.
    """
    def __init__(self, fname):
        """ Initialize an Image class object from an iamge file.

        Parameters
        ----------
        fname : str
            The filename of a valid image file.
        """
        ## Open image
        self.fname = fname
        self._image = cv2.imread(fname)
        if self._image is None:
            print("'{}' is not a valid image file.".format(fname))
            sys.exit()

        ## Properties of the original image
        self._height, self._width = self._image.shape[:2]
        self._aspect = float(self._height) / self._width

        ## Zoom
        self._zoom = 1
        self._zoom_x = self._width/2.0
        self._zoom_y = self._height/2.0

        ## Canvas properties used to generate the view
        self._canvas_width = None
        self._canvas_height = None


    def _set_zoom_center(self,center):
        """ Set the center pixel to be shown in a zoomed view.

        Parameters
        ----------
        center : tuple(int,int)
            (x,y) coordinates of original image.
        """
        x,y = center
        x = max(0, x)
        x = min(x, self._width)
        y = max(0, y)
        y = min(y, self._height)
        self._zoom_x = x
        self._zoom_y = y

    def get_zoom(self):
        """ Get the current zoom factor.

        Returns
        -------
        float
            The zoom factor.
        """
        return self._zoom

    def set_zoom(self,zoom):
        """ Set the zoom factor.

        Parameters
        ----------
        zoom : float
            The zoom factor.
        """
        if zoom < 1.0:
            zoom = 1.0
        self._zoom = zoom

    def reset_zoom(self):
        """ Reset the zoom to show the full image. """
        self._zoom = 1.0
        self._zoom_x = self._width/2
        self._zoom_y = self._height/2

    def move_right(self):
        """ Adjust the image view by moving the window right 10%. """
        zoom_step = 0.1 * self._width/self._zoom
        self._set_zoom_center( (self._zoom_x + zoom_step, self._zoom_y) )

    def move_left(self):
        """ Adjust the image view by moving the window left 10%. """
        zoom_step = 0.1 * self._width/self._zoom
        self._set_zoom_center( (self._zoom_x - zoom_step, self._zoom_y) )

    def move_up(self):
        """ Adjust the image view by moving the window up 10%. """
        zoom_step = 0.1 * self._height/self._zoom
        self._set_zoom_center( (self._zoom_x, self._zoom_y - zoom_step) )

    def move_down(self):
        """ Adjust the image view by moving the window down 10%. """
        zoom_step = 0.1 * self._height/self._zoom
        self._set_zoom_center( (self._zoom_x, self._zoom_y + zoom_step) )

    def _generate_view(self, canvas_shape=None, zoom=None, center=None, splitcell=False, cellsize=None, rgb=True):
        """ Return a resized version of the original image such that it fits within the canvas.

        Parameters
        ----------
        canvas_shape : tuple(int,int)
            (width,height) of the canvas where the image is supposed to be displayed.
        zoom : int
            Zoom factor. No zoom means zoom=1.
        center : tuple(int,int)
            (x,y) coordinates of the focus of the zoom.
        splitcell : bool, optional
            Whether each character cell should be treated as two pixels (default: False).
        cellsize : tuple(int,int)
            (height,width) Number of pixels per cell.
        rgb : bool, optional
            If False, return a grayscale image (default: True).

        Returns
        -------
        numpy.array
            An opencv compatible array representing the image.
        """
        ##
        ## Step 1. Determine the actual width and height of the desired image.
        ##
        if canvas_shape is not None:
            self._canvas_width, self._canvas_height = canvas_shape
        if zoom is not None:
            self._zoom = zoom
        if center is not None:
            self._set_zoom_center(center)

        ##
        ## Adjust the width and height to conform with the aspect ratio.
        ##
        height = self._canvas_height
        width = self._canvas_width
        aspect = self._aspect / FONT_ASPECT
        if float(self._canvas_height)/self._canvas_width > aspect:
            height = int(np.around(aspect * self._canvas_width))
        else:
            width = int(np.around(self._canvas_height / aspect))

        ##
        ## Zoom
        ##
        _w = self._width/float(self._zoom)
        _h = self._height/float(self._zoom)
        _x1 = int(self._zoom_x - _w/2)
        _x2 = int(self._zoom_x + _w/2)
        _y1 = int(self._zoom_y - _h/2)
        _y2 = int(self._zoom_y + _h/2)

        self._zoom_x = _x1 + _w/2
        self._zoom_y = _y1 + _h/2

        ##
        ## Check if the zoom changes the aspect ratio
        ##
        # Image pixels per cell:
        scale_x = (_x2 - _x1) / float(self._canvas_width)
        scale_y = (_y2 - _y1) / float(self._canvas_height)
        # Cut-off pixels:
        cutoff_x = self._width - (_x2 - _x1)
        cutoff_y = self._height - (_y2 - _y1)
        # Available space:
        available_x = self._canvas_width - width
        available_y = self._canvas_height - height

        if available_x > 0:
            # Append pixels
            append_x = min(cutoff_x, available_x * scale_x)
            _x1 -= int(math.ceil(append_x/2.0))
            _x2 += int(math.floor(append_x/2.0))
            # Adjust width
            width = min(
                int(width + append_x/scale_x),
                self._canvas_width
            )
        elif available_y > 0:
            # Append pixels
            append_y = min(cutoff_y, available_y * scale_y)
            _y1 -= int(math.ceil(append_y/2.0))
            _y2 += int(math.floor(append_y/2.0))
            # Adjust height
            height = min(
                int(height + append_y/scale_y),
                self._canvas_height
            )

        ##
        ## Check boundaries
        ##
        if _x1 < 0:
            _x2 -= _x1
            _x1 = 0
        if _x2 > self._width:
            _x1 -= (_x2 - self._width)
            _x2 = self._width
        if _y1 < 0:
            _y2 -= _y1
            _y1 = 0
        if _y2 > self._height:
            _y1 -= (_y2 - self._height)
            _y2 = self._height

        image = self._image[_y1:_y2,_x1:_x2,:]

        ##
        ## Account for split character cells
        ##
        if splitcell:
            height *= 2

        if cellsize is not None:
            width *= cellsize[1]
            height *= cellsize[0]

        ##
        ## Resize
        ##
        image = cv2.resize(image, (width,height))

        ##
        ## RGB or Grayscale?
        ##
        if rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image


    ########################################################################################
    # Conversion to ASCII or color #########################################################
    ########################################################################################

    def _to_ascii(self):
        """ (deprecated) Render an ASCII representation of the image. """
        gray_image = self._generate_view(rgb=False)
        # Convert to ASCII
        asciiize = np.vectorize(gray2char)
        chars = asciiize(gray_image)
        color_black = np.zeros(chars.shape, dtype=np.uint8)
        color_white = np.ones(chars.shape, dtype=np.uint8) * 231
        return np.stack([color_black,color_white,chars],axis=-1)

    def _to_highres(self):
        """ (deprecated) Render a resolution-optimized representation of the image. """
        rgb_image = self._generate_view(splitcell=True)
        if FLAG_MP:
            upper = parallel_apply_along_axis(rgb_closest,2,rgb_image[0::2,:,:])
            lower = parallel_apply_along_axis(rgb_closest,2,rgb_image[1::2,:,:])
        else:
            upper = np.apply_along_axis(rgb_closest,2,rgb_image[0::2,:,:])
            lower = np.apply_along_axis(rgb_closest,2,rgb_image[1::2,:,:])

        chars = np.chararray(upper.shape, unicode=True)
        chars[:] = u'\u2584'
        return np.stack([upper,lower,chars],axis=-1)

    def _to_fast_color(self):
        """ Render an optimal colored representation of the image.
        This method is a trade-off between _to_color() and _to_highres()
        """
        rgb_image = self._generate_view(splitcell=True)
        upper = rgb_image[0::2,:,:]
        lower = rgb_image[1::2,:,:]
        concat = np.concatenate((upper,lower), axis=2)
        if FLAG_MP:
            result = parallel_apply_along_axis(best_representation,2,concat)
        else:
            result = np.apply_along_axis(best_representation,2,concat)
        return result

    def _to_color(self):
        """ Render an optimal colored representation of the image. """
        rgb_image = self._generate_view(cellsize=CELLSIZE)
        h,w,_ = rgb_image.shape
        cells = np.array(np.split(
            np.array(np.split(
                rgb_image, int(w/CELLSIZE[1]), axis=1
            )), int(h/CELLSIZE[0]), axis=1
        ))
        cells = cells.reshape(cells.shape[0], cells.shape[1], CELLSIZE[0]*CELLSIZE[1]*3)
        if FLAG_MP:
            result = parallel_apply_along_axis(pixels2cell,2,cells)
        else:
            result = np.apply_along_axis(pixels2cell,2,cells)
        return result

    def _to_edges(self):
        """ (Experimental) Render an edge-detection based representation of the image.
        """
        # Split channels.
        image = self._generate_view()
        channels = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        # edges = cv2.Canny(channels[1],200,300)
        edges = cv2.Canny(channels[2],100,250)
        # edges = cv2.Canny(channels[0],300,400) / 255.0 + \
        # edges = cv2.Canny(channels[1],200,300) / 255.0 + \
        #     cv2.Canny(channels[2],100,200) / 255.0

        edges = np.array((edges > 1.0), dtype=np.uint8)

        kernel = np.array([[  1,   2,   4],
                           [  8,   0,  16],
                           [ 32,  64, 128]])

        # edge_chars = [
        #     ('-',  [8,16,24,20,28]),
        #     ('|',  [2,64,66]),
        #     ('\\', [1,128,129,137,145]),
        #     ('/',  [4,32,36]),
        #     (',',  [127]),
        #     ('`',  [3,9,17]),
        #     ('\'', [6,12,14]),
        #     ('v',  [5,7,88]),
        #     ('>',  [33,41,82]),
        #     ('<',  [74,132,148]),
        #     ('^',  [26,160,224])]

        edge_chars = [
            (u'\u2500', [8,16,24,20,28]),               # ─
            (u'\u2502', [2,64,66,194,98,70]),           # │
            (u'\u2572', [1,128,129,137,145,147,201]),   # ╲
            (u'\u2571', [4,32,36,44,52,126,46,116]),    # ╱
            (u'\u250C', [80,48]),                       # ┌
            (u'\u2510', [72,136]),                      # ┐
            (u'\u2514', [18,17]),                       # └
            (u'\u2518', [10,12]),                       # ┘
            (u'/',      [68,84,34,42]),                 # /

            (u'v',      [5,7,88]),
            (u'>',      [33,41,82]),
            (u'<',      [74,132,148]),
            (u'^',      [26,160,224]),
        ]

        def assign_edge_char(value):
            for c, vals in edge_chars:
                if value in vals:
                    return c
            return u'o'
        edgize = np.vectorize(assign_edge_char)
        dst = cv2.filter2D(edges,-1,kernel)
        edged = edgize(dst)
        edged[edges==0] = u' '
        color_black = np.zeros(edged.shape, dtype=np.uint8)
        color_white = np.ones(edged.shape, dtype=np.uint8) * 231
        return np.stack([color_black,color_white,edged],axis=-1)

    ########################################################################################
    # Rendering to screen ##################################################################
    ########################################################################################

    def render(self, shape=None, mode='ascii', zoom=None, center=None):
        """ Render a representation of the image with the desired width, height and mode.

        Parameters
        ----------
        shape : tuple(int,int), optional
            (width,height) of the canvas.
        mode : str
            The image representation mode. One of 'ascii', 'color', 'highres', 'optimal', 'edge'.
        zoom : float, optional
            The zoom factor.

        Returns
        -------
        numpy.array
            A numpy.array object that contains background color, foreground color and character for
            each pixel in the output image.
        """
        if shape is not None:
            self._canvas_width, self._canvas_height = shape
        if zoom is not None:
            self._zoom = zoom
        if center is not None:
            self._set_zoom_center(center)

        if mode == 'color':
            image = self._to_color()
        if mode == 'fast':
            image = self._to_fast_color()
        if mode == 'highres':
            image = self._to_highres()
        elif mode == 'ascii':
            image = self._to_ascii()
        elif mode == 'edge':
            image = self._to_edges()
        return image
