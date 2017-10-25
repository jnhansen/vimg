import os
import sys
import vimg

def main():
    if len(sys.argv) < 2:
        print('Usage: vimg path/to/image'.format(sys.argv[0]))
        sys.exit()
    f = os.path.expanduser(sys.argv[1])
    image = vimg.Image(f)
    gui = vimg.GUI(image, mode='optimal')
    gui.main()

if __name__ == '__main__':
    main()
