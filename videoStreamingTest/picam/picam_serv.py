import io
import socket
import struct
import time
import picamera


server_socket = socket.socket()
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #Allow the address to be used again
server_socket.bind((socket.gethostname(), 8000))
server_socket.listen(0)

print("Waiting for connection")
## Make a file-like object out of the connection
connection = server_socket.accept()[0].makefile('wb')


try:
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    # Start a preview and let the camera warm up for 2 seconds
#    camera.start_preview()
    time.sleep(2)

    # Note the start time and construct a stream to hold image data
    # temporarily (we could write it directly to connection but in this
    # case we want to find out the size of each capture first to keep
    # our protocol simple)
    start = time.time()
    stream = io.BytesIO()
    
    t0 = time.time()
    for foo in camera.capture_continuous(stream, 'jpeg', use_video_port = True):
        # Write the length of the capture to the stream and flush to
        # ensure it actually gets sent
        connection.write(struct.pack('<L', stream.tell()))
        connection.flush()
        # Rewind the stream and send the image data over the wire
        stream.seek(0)
        connection.write(stream.read())
        # If we've been capturing for more than 30 seconds, quit
        if time.time() - start > 30:
            break
        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()
        t1 = time.time()
        print(1/(t1-t0))
        t0 = t1
    # Write a length of zero to the stream to signal we're done
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    server_socket.close()