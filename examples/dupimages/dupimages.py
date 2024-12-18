"""
Finds duplicate images using perceptual hash
Creates hask.pkl cache files in each subdirectory for speed
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2015 Philippe Guglielmetti"
__license__ = "All rights reserved"
__version__ = '$Id$'

import os
import logging
import sys
import pickle
from multiprocessing import Pool, TimeoutError


from goulib.image import Image
from goulib import decorators


def hashdir(dirName, subdirList, fileList):
    logging.debug('start %s' % dirName)

    def ext(fname):
        return fname[-3:] in ['jpg', 'peg', 'png', 'gif', 'bmp']

    images = list(filter(ext, fileList))

    d = {}
    if not fileList:
        return dirName, d

    pklfile = 'hash.pkl'
    if pklfile in fileList:
        pkl = open(dirName + '\\' + pklfile, 'rb')
        logging.debug('%s : reading hash.pkl' % dirName)
        d = pickle.load(pkl)  # local dic
        pkl.close()
        if len(d) == len(images):
            return dirName, d
        else:
            os.remove(r'%s\hash.pkl' % dirName)

    logging.info('%s : building hash.pkl' % dirName)
    for fname in images:
        file = '%s\%s' % (dirName, fname)

        try:
            img = Image(file)
            h = hash(img)
        except Exception as e:
            logging.error('%s : %s' % (file, e))
            h = 0

        if h in d:  # (almost) duplicate in same dir
            d[h].append(fname)
        else:
            d[h] = [fname]
    if d:
        logging.debug('%s : SAVING hash.pkl' % dirName)
        pkl = open(r'%s\hash.pkl' % dirName, 'wb')
        pickle.dump(d, pkl)
        pkl.close()
    return dirName, d


def purge(a, b):
    a = Image(a)
    b = Image(b)
    if a < b:
        logging.info('delete smaller %s', a.path)
        os.remove(a.path)
        return b
    if a > b:
        logging.info('delete smaller %s', b.path)
        os.remove(b.path)
        return a
    if a.path < b.path:
        logging.info('delete %s', a.path)
        os.remove(a.path)
        return b
    else:
        logging.info('delete %s', b.path)
        os.remove(b.path)
        return a


def callback(r):
    # merge local to global
    dirName, d = r
    logging.debug('callback %s' % dirName)
    try:
        for h in d:
            a = r'%s\%s' % (dirName, d[h][0])
            a = os.path.basename(a)
            d[h] = [a]
            if h in dic:
                dic[h].append((dirName, a))
            else:
                dic[h] = [(dirName, a)]

            n = len(dic[h])
            if n > 1:  # duplicates in different dirs
                logging.info('%d duplicates found' % n)

    except Exception as e:
        logging.error(e)


if __name__ == '__main__':

    rootDir = r'D:\Users\Philippe\Images'

    pool = Pool(processes=5)

    logging.root.setLevel(logging.WARNING)

    dic = {}

    i = 0
    for params in os.walk(rootDir, topdown=False):
        i = i+1
        pool.apply_async(hashdir, params, callback=callback,
                         error_callback=logging.error)
    logging.info('%d dirs scanned' % i)

    pool.close()
    pool.join()

    # if we didn't interrupt, rewrite the results more comprehensively
    out = open('dups.txt', 'w')

    for h in dic:
        n = len(dic[h])
        if n > 1:  # duplicates in different dirs
            for (dir, file) in dic[h]:
                path = r'%s\%s' % (dir, file)
                out.write('%s\n' % path)
            out.write('\n')
            out.flush()

    dirs = {}

    # generate a dir oriented list
    for h in dic:
        n = len(dic[h])
        if n > 1:  # duplicates in different dirs
            for (dir, file) in dic[h]:
                if not dir in dirs:
                    # dirs with the same h were created at the same moment
                    dirs[dir] = [h]
                dirs[dir].append(file)

    prevh = None
    for dir in dirs:
        h = dirs[dir][0]
        if h != prevh:
            if prevh is not None:
                out.write('</table></body></html>\n')
                out.close()
            prevh = h
            name = str.format('{:03d}-{:08X}', (h, len(dirs[dir])))
            out = open('out/%s.htm' % name, 'w')
            out.write('<html><body><table>\n')
        out.write('<tr>')
        for file in dirs[dir][1:]:
            path = r'%s\%s' % (dir, file)
            out.write('<td><a href="%s">%s</br>' % (path, path))
            out.write('<img src="%s" width="320px"></a></td>' % path)
        out.write('</tr>\n')
    out.write('</table></body></html>\n')
    out.close()
