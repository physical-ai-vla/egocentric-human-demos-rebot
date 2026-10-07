#!/bin/bash
# usage: snap.sh NAME  -> saves NAME_{left,middle,right}.jpg + NAME.json (joints) and reports chessboard detection on the right wrist
curl -s -m 10 localhost:8021/observe -o $1.json && ~/xvla-mac/bin/python -c "
import json,base64,cv2,numpy as np;o=json.load(open('$1.json'))
for k,v in o['images'].items(): open('$1_'+k+'.jpg','wb').write(base64.b64decode(v))
g=cv2.imread('$1_right.jpg',0); ok,c=cv2.findChessboardCorners(g,(8,5),flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE)
print('$1 board found' if ok else '$1 board NOT found', '' if not ok else np.round(c.reshape(-1,2).mean(0)))"
