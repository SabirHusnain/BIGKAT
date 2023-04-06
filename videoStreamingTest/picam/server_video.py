# -*- coding: utf-8 -*-
"""
"""

import socket
import time
import cv2
import numpy as np
import struct
import io



server_socket = socket.socket()
server_socket.bind((socket.gethostname(), 8000))
server_socket.listen()

print("Waiting for connection")
## Make a file-like object out of the connection
connection = server_socket.accept()[0].makefile('wb')




cap = cv2.VideoCapture(0)

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True: 
        
        res, imencode = cv2.imencode('.jpeg', frame, encode_param)
   
        data = imencode.tostring()
        connection.write(struct.pack('<L', len(data)))
        connection.flush()
           
     
        connection.write(data)  
        
 
    else:
        break
    
connection.close()
# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
#