# -*- coding: utf-8 -*-
"""
"""

import socket
import time
import cv2
import io
import struct
import numpy as np



# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
client_socket = socket.socket()
client_socket.connect((socket.gethostname(), 8000))

connection = client_socket.makefile('rb')

t0 = time.time()
while True: 
    
    image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
    if not image_len:
        break

    frame = connection.read(image_len)
    
    
    # Construct a numpy array from the stream
    
    data = np.fromstring(frame, dtype=np.uint8)

    # "Decode" the image from the array, preserving colour
    image = cv2.imdecode(data, 1)
    
    t1 = time.time()
    print("Time: {}".format(t1-t0))
    t0 = t1
  
    if image != None:
#        print(image.shape)
        cv2.imshow("WEB", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

  