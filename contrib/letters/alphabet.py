#! /usr/bin/env python
'''
Generates a data file for alphabets and saves it in a matlab file.

@author: Tomoki Tsuchida <ttsuchida@ucsd.edu>
'''
from numpy import *
from PIL import Image, ImageDraw, ImageFont
from scipy.io import savemat

def main(p,k):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456'
    font = ImageFont.truetype('slkscr.ttf', p)
    L = zeros((p*p, k))
    
    l = 0
    li = 0
    sgn = 1
    while l < k:
        letter = letters[li]
        li += 1
        img = Image.new('L', (p,p))
        data = ImageDraw.Draw(img)
        data.fontmode = '1'
        data.text((0,0), letter, fill=255, font=font)
        L[:,l] = sgn*array(img).reshape((p*p,))
        l += 1
        if k > len(letters):
            L[:,l] = sgn*array(img).T.reshape((p*p,))
            l += 1
        sgn *= -1
    
    savemat("letters-%d-%d.mat" % (k, p), {'D' : L })
    print "Created %d %d-by-%d letters" % (k, p, p)

if __name__ == '__main__':
    import os, sys
    os.chdir(os.path.dirname(sys.argv[0]))

    import argparse
    
    parser = argparse.ArgumentParser(description='Creates matlab file for letters.')
    parser.add_argument('p', type=int, default=8, metavar='p', help='creates p-by-p images')
    parser.add_argument('k', type=int, default=25, metavar='k', help='creates k dictionary elements')

    args = parser.parse_args()
    main(**vars(args))