vimg: A CLI image viewer.
#########################

Author: Johannes Hansen

Why?
====
If you are like me, you spend a lot of time in command line environments. One of the few things that
are hard to deal with in such an environment is images.

This little tool is meant to allow for quick viewing of image files in the command line.

The image is rendered using a combination of background color, foreground color and unicode character
for each character cell to optimally represent the original pixels. The challenge is the limited
color palette of 256 colors.

In the simplest case, each character cell represents one pixel. However, the resolution can be doubled
by printing a half-block character in a different color in that character cell.
Alternatively, the color accuracy can be improved by mixing two available colors in foreground and background,
thus losing the gained resolution.

The default mode attempts to optimize the rendering by optimizing resolution in areas of high
detail, and optimizing color accuracy in areas of low detail.

Installation
============
.. code-block:: bash

   $ pip install vimg


Requirements
------------
A terminal that supports 256 colors.

The script is based on ``curses`` and ``opencv`` for Python. Make sure you have those installed.

Usage
=====
.. code-block:: bash

    $ vimg path/to/image



GUI modes
=========
Once in the GUI, you can change between different viewing modes:


============= ========= ====================================================================
Key shortcut  Mode      Description
============= ========= ====================================================================
``c``           color     display color-optimized image
``h``           highres   display resolution optimized image
``o``           optimal   (default) display an optimal representation of the image. Trade-off between coloraccuracy and resolution
``a``           ascii     display a black-and-white representation of the image
``e``           edge      (experimental) edge detection based rendering
``r``           --        refresh the screen
``q``           --        quit
============== ========= ====================================================================

Limitations
===========
The script currently only supports image files that are natively supported by OpenCV (``.jpg``, ``.png``, ``.bmp``).


To Do
=====
Future plans include:

* Support for more image file types, e.g. ``.gif``
* Improvement of the edge detection mode
