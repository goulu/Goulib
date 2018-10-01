#!/usr/bin/python
# -*- coding: utf-8 -*-

import os,logging, sys, pickle
from Goulib.image import Image

def show(img,title='image'):
    cv2.imshow(title,img)
    cv2.waitKey(1)
    return img

def purge(a,b):
    a=Image(a)
    b=Image(b)
    if a<b:
        logging.info('delete smaller %s',a)
        os.remove(a.path)
        return b
    if a>b:
        logging.info('delete smaller %s',b)
        os.remove(b.path)
        return a
    if a.path<b.path:
        logging.info('delete %s',a)
        os.remove(a.path)
        return b
    else:
        logging.info('delete %s',b)
        os.remove(b.path)
        return a
    
def directory(dirName, fileList):
    pklfile='hash.pkl'
    if pklfile in fileList:
        pkl = open(dirName+'\\'+pklfile, 'rb')
        logging.debug('%s : reading hash.pkl'%dirName)
        d = pickle.load(pkl) #local dic
        pkl.close()
        return d,False
    else:
        logging.debug('%s : building hash.pkl'%dirName)
        d={}
        for fname in fileList:
            ext=fname[-3:]
            if ext not in 'jpg': continue
            file='%s\%s'%(dirName,fname)
            try:
                img = Image(file)
            except:
                logging.warning('could not read %s'%file)
                continue
            try:
                h=hash(img)
            except:
                logging.warning('could not hash %s'%file)
                continue 
            if h in d: # (almost) duplicate in same dir
                d[h].append(fname)
            else:
                d[h]=[fname]
        return d,True

    
logging.basicConfig(level=logging.INFO)
 
rootDir = r'.'

dic={} # global dic
out=open('dups.txt','w')
for dirName, subdirList, fileList in os.walk(rootDir, topdown=False):
    if not fileList: continue
    d,save=directory(dirName, fileList)
    #merge local to global
    for h in d:
        a=r'%s\%s'%(dirName,d[h][0])
        for b in d[h][1:]: #duplicates in same dir
            try:
                a=purge(a,r'%s\%s'%(dirName,b)).path
            except PermissionError:
                continue #TODO: something more clever...
            finally:
                save=True
        a=os.path.basename(a)
        d[h]=[a]
        if h in dic:
            dic[h].append((dirName,a))
        else:
            dic[h]=[(dirName,a)]
            
        if len(dic[h])>1: #duplicates in different dirs
            logging.info('')
            out.write('\n')
            for (dir,file) in dic[h]:
                img = Image(r'%s\%s'%(dir,file))
                logging.info(img)
                out.write('%s\n'%img.path)
            out.flush()
            
    if save:
        pkl = open(r'%s\hash.pkl'%dirName, 'wb')
        pickle.dump(d,pkl)
        pkl.close()



    