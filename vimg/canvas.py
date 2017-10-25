import sys
import curses
from threading import Timer

PY2 = sys.version_info < (3,0)

class GUI:
    """ An abstract class representing the Terminal GUI.
    """
    def __init__(self, image, mode='ascii'):
        """ Initialize the GUI with an Image object.

        Parameters
        ----------
        image : Image
            An Image class object to be visualized in the GUI.
        mode : str, optional
            The representation mode of the image. (default: ascii)
        """
        self.image = image
        self.mode = mode
        self.WINDOW = curses.initscr()

        self.OFFSET_Y = (1,1)
        self.OFFSET_X = (0,0)

        self.MOVE_CURSOR = curses.tigetstr("cup")
        self.SET_FG = curses.tigetstr("setaf")
        self.SET_BG = curses.tigetstr("setab")
        self.CLEAR_SCREEN = curses.tigetstr("clear").decode("utf-8")

        self.DEFAULT_FG_COLOR = 9
        self.DEFAULT_BG_COLOR = 0
        self.ZOOM_DELAY = 0.2

        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.WINDOW.keypad(1)
        curses.mousemask(1)

        curses.start_color()
        curses.use_default_colors()

        # self._zoom = 1
        # self._zoom_center = (self.width / 2.0, self.height / 2.0)

        self._fg_color = self.DEFAULT_FG_COLOR
        self._bg_color = self.DEFAULT_BG_COLOR
        self._cursor_x = 0
        self._cursor_y = 0

        self._render_timer = None

        self._stdout = ''

        self.clear()

    @property
    def term_height(self):
        """ Height of the terminal window. """
        return self.WINDOW.getmaxyx()[0]

    @property
    def term_width(self):
        """ Width of the terminal window. """
        return self.WINDOW.getmaxyx()[1]

    @property
    def shape(self):
        """ (width,height) of the terminal window. """
        return (self.width, self.height)

    @property
    def height(self):
        """ Height of the canvas for drawing the image. """
        return self.WINDOW.getmaxyx()[0] - sum(self.OFFSET_Y)

    @property
    def width(self):
        """ Width of the canvas for drawing the image. """
        return self.WINDOW.getmaxyx()[1] - sum(self.OFFSET_X)

    def set_zoom(self,zoom):
        """ Wrapper function for Image.set_zoom(). """
        self.image.set_zoom(zoom)

    def get_zoom(self):
        """ Wrapper function for Image.get_zoom(). """
        return self.image.get_zoom()

    def _write(self, text):
        try:
            sys.stdout.write(text)
        except IOError:
            pass

    def _print(self, text, x, y, color, bg, attr=None):
        """ Generate a binary string representing a colored text at a given position.

        Parameters
        ----------
        text : str
            The text to print.
        x : int
            The x position of the leftmost character of the text.
        y : int
            The y position of the text.
        color : int
            The foreground color of the text (xterm-256 color).
        bg : int
            The background color of the text (xterm-256 color).
        attr : int
            Not implemented

        Returns
        -------
        A binary string that can be printed directly to stdout.
        """
        cursor = u''
        if x != self._cursor_x or y != self._cursor_y:
            cursor = curses.tparm(self.MOVE_CURSOR, y, x).decode('utf-8')
            self._cursor_x = x + len(text)
            self._cursor_y = y

        fg_color = u''
        if color != self._fg_color:
            fg_color = curses.tparm(self.SET_FG, color).decode('utf-8')
            self._fg_color = color

        bg_color = u''
        if bg != self._bg_color:
            bg_color = curses.tparm(self.SET_BG, bg).decode('utf-8')
            self._bg_color = bg

        fmt = cursor + fg_color + bg_color
        if PY2 or not isinstance(text, str):
            text = text.decode('utf-8')
        return fmt + text

    def printat(self, *args, **kwargs):
        """ Like _print, but append to internal buffer. """
        self._stdout += self._print(*args, **kwargs)

    def _print_statusline(self):
        """ Draw the very simple top and bottom statuslines of the GUI. """
        statusline_top = self._print(
            ('{:%s}'%self.term_width).format(' {} ({}x{})'.format(
                self.image.fname, self.image._image.shape[1], self.image._image.shape[0],
                # int(100 * self.image.w / self.image._image.shape[1])
            )),
            0,0,
            curses.COLOR_WHITE, 0
        )
        statusline_bottom = self._print(
            ('{:%s}'%self.term_width).format(' A:ascii  C:color  H:highres  O:optimal  E:edges  r:refresh  q:quit  +/-:zoom  hjkl:move'),
            0,self.term_height-1,
            curses.COLOR_WHITE, 0
        )
        self._write(statusline_top)
        self._write(statusline_bottom)
        self._set_default_colors()
        sys.stdout.flush()

    def _loading_screen(self):
        """ Not implemented. """
        pass

    def _set_default_colors(self):
        self._write( curses.tparm(self.SET_FG, self.DEFAULT_FG_COLOR).decode("utf-8") )
        self._write( curses.tparm(self.SET_BG, self.DEFAULT_BG_COLOR).decode("utf-8") )
        self._fg_color = self.DEFAULT_FG_COLOR
        self._bg_color = self.DEFAULT_BG_COLOR

    def clear(self):
        """ Clear screen content. """
        self._write(self.CLEAR_SCREEN)
        self._set_default_colors()
        sys.stdout.flush()

    def clear_buffer(self):
        """ Clear screen content buffer. """
        self._stdout = ''

    def output(self):
        """ Print the content buffer to screen. """
        self._print_statusline()
        self._write(self._stdout)
        self._set_default_colors()
        try:
            sys.stdout.flush()
        except IOError:
            pass

    def refresh(self):
        """ Refresh the screen from the content buffer. """
        self.clear()
        self.output()

    def save(self):
        """ Save the content buffer to a file.

        The file can later be viewed in the terminal using cat
        """
        with open('stdout.log','w') as f:
            f.write(self._stdout + '\n')

    def quit(self):
        """ End curses application. """
        curses.nocbreak()
        curses.echo()
        curses.curs_set(1)
        curses.mousemask(0)
        curses.endwin()

    def render(self, refresh=True):
        """ Render the image and print to screen. """
        im = self.image.render(shape=self.shape, mode=self.mode)
        xpos = int((self.width - im.shape[1])/2)
        ypos = int((self.height - im.shape[0])/2)
        self.clear_buffer()
        self.clear()

        for y,line in enumerate(im):
            for x,vec in enumerate(line):
                bg, fg, char = vec
                bg = int(bg)
                fg = int(fg)
                self.printat(char, x+xpos+self.OFFSET_X[0], y+ypos+self.OFFSET_Y[0], color=fg, bg=bg)

        if refresh:
            self.refresh()

    def getch(self):
        """ Get character input. """
        return self.WINDOW.getch()

    def _schedule_render(self,seconds):
        """ Start a timed call to self.render(). """
        if self._render_timer is not None:
            self._render_timer.cancel()
        def wrapper():
            self.render()
            self._render_timer = None
        self._render_timer = Timer(seconds, wrapper)
        self._render_timer.start()

    def _screen2pixel(self,x,y):
        """ Not implemented. """
        pass

    def main(self):
        """ The main loop of the GUI. Handles keyboard input. """
        try:
            self.render(refresh=False)
            key_map = {
                ord('C') : 'color',
                ord('H') : 'highres',
                ord('O') : 'optimal',
                ord('A') : 'ascii',
                ord('E') : 'edge'
            }
            ##
            ## Pass an initial key press to make sure the program paints the
            ## screen at startup
            ##
            curses.ungetch(ord('r'))
            while True:
                c = self.getch()
                if c == curses.KEY_RESIZE:
                    ##
                    ## Redrawing in ASCII is fast. Color takes longer, therefore
                    ## user longer interval
                    ##
                    if self.mode == 'ascii':
                        self._schedule_render(.3)
                    else:
                        self._schedule_render(1.)
                elif c in key_map:
                    self.mode = key_map[c]
                    self.render()
                elif c == ord('r'):
                    self.refresh()
                elif c == ord('s'):
                    self.save()
                elif c == ord('+'):
                    self.image.set_zoom(self.get_zoom() * 1.3)
                    self._schedule_render(self.ZOOM_DELAY)
                elif c == ord('-'):
                    self.image.set_zoom(self.get_zoom() / 1.3)
                    self._schedule_render(self.ZOOM_DELAY)
                elif c == ord('0'):
                    self.image.set_zoom(1.0)
                    self.render()
                elif c == curses.KEY_DOWN or c == ord('j'):
                    self.image.move_down()
                    self._schedule_render(self.ZOOM_DELAY)
                elif c == curses.KEY_UP or c == ord('k'):
                    self.image.move_up()
                    self._schedule_render(self.ZOOM_DELAY)
                elif c == curses.KEY_LEFT or c == ord('h'):
                    self.image.move_left()
                    self._schedule_render(self.ZOOM_DELAY)
                elif c == curses.KEY_RIGHT or c == ord('l'):
                    self.image.move_right()
                    self._schedule_render(self.ZOOM_DELAY)
                elif c == ord('q'):
                    break

        except Exception as e:
            ##
            ## Make sure to quit curses mode before any exception is raised.
            ##
            self.quit()
            raise
        except KeyboardInterrupt:
            self.quit()

        self.quit()
