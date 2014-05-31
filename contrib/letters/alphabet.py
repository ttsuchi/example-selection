#! /usr/bin/env python
'''
Generates a data file for alphabets and saves it in a matlab file.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from PIL import Image, ImageDraw, ImageFont
from scipy.io import savemat

def main(p):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456'
    font = ImageFont.truetype('slkscr.ttf', p)
    L = zeros((p*p, len(letters)*2))
    
    l = 0
    sgn=1
    for letter in letters:
        img = Image.new('L', (p,p))
        data = ImageDraw.Draw(img)
        data.fontmode = '1'
        data.text((0,0), letter, fill=255, font=font)
        L[:,l] = sgn*array(img).reshape((p*p,))
        l += 1
        L[:,l] = sgn*array(img).T.reshape((p*p,))
        l += 1
        sgn *= -1
    
    savemat("alphabet-%d.mat" % p, {'D' : L })
    print "Created %d-by-%d letters" % (p, p)

if __name__ == '__main__':
    import os, sys
    os.chdir(os.path.dirname(sys.argv[0]))

    import argparse
    
    parser = argparse.ArgumentParser(description='Creates matlab file for letters.')
    parser.add_argument('p', type=int, default=8, metavar='p', help='creates p-by-p images')

    args = parser.parse_args()
    main(**vars(args))