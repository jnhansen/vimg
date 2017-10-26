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

In the simplest case, each character cell represents one pixel. However, the resolution can be
increased by printing unicode characters that better capture the structure of the image.
Alternatively, the color accuracy can be improved by mixing two available colors in foreground and
background, thus losing the gained resolution.

The default mode attempts to optimize the rendering by optimizing resolution in areas of high
detail, and optimizing color accuracy in areas of low detail.

Installation
============
.. code-block:: bash

   $ pip install vimg


Requirements
------------
A terminal that supports 256 colors.

The script is based on ``curses`` and ``opencv`` for Python.

Usage
=====
.. code-block:: bash

    $ vimg path/to/image


GUI modes
=========
Once in the GUI, you can change between different viewing modes:

+--------------------------+---------+-------------------------------------------------------------+
| Key shortcut             |  Mode   |  Description                                                |
+==========================+=========+=============================================================+
| ``c``                    | color   | (default) display optimal representation of image           |
+--------------------------+---------+-------------------------------------------------------------+
| ``f``                    | fast    | display image in fast mode (reduced resolution)             |
+--------------------------+---------+-------------------------------------------------------------+
| ``a``                    | ascii   | display a black-and-white representation of the image       |
+--------------------------+---------+-------------------------------------------------------------+
| ``e``                    | edge    | (experimental) edge detection based rendering               |
+--------------------------+---------+-------------------------------------------------------------+
| ``+``/``-``              | --      | zoom in/out (by 30%)                                        |
+--------------------------+---------+-------------------------------------------------------------+
| | ``h`` ``j`` ``k`` ``l``| --      | move view (by 10%)                                          |
| | or arrow keys          |         |                                                             |
+--------------------------+---------+-------------------------------------------------------------+
| ``0``                    | --      | reset zoom                                                  |
+--------------------------+---------+-------------------------------------------------------------+
| ``r``                    | --      | refresh the screen                                          |
+--------------------------+---------+-------------------------------------------------------------+
| ``q``                    | --      | quit                                                        |
+--------------------------+---------+-------------------------------------------------------------+


Notes
=====
The results will be better if you use a font that correctly displays unicode block element characters
with the full line height, such as DejaVu Sans.

Limitations
===========
The script currently only supports image files that are natively supported by OpenCV (``.jpg``,
``.png``, ``.bmp``).


To Do
=====
Future plans include:

* Support for more image file types, e.g. ``.gif``
* Improvement of the edge detection mode
* Make ``opencv`` dependency optional
* Improve color gradients at contrast-rich edges
