from __future__ import print_function
import sys
sys.path.append("..")
import utils
from utils.app_utils import *
from utils.objDet_utils import *

from utils.objDet_utils import worker, crop_worker
import argparse
import multiprocessing
from multiprocessing import Queue, Pool
import cv2
import os
sys.path.append('/home/frosted007/Documents/Cargo_Scan/utils')
import objDet_utils
from objDet_utils import detect_objects


def realtime(args):
    """
    Read and apply object detection to input real time stream (webcam)
    """

    # If display is off while no number of frames limit has been define: set diplay to on
    if((not args["display"]) & (args["num_frames"] < 0)):
        print("\nSet display to on\n")
        args["display"] = 1

    # Set the multiprocessing logger to debug if required
    if args["logger_debug"]:
        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)

    # Multiprocessing: Init input and output Queue and pool of workers
    input_q = Queue(maxsize=args["queue_size"])
    output_q = Queue(maxsize=args["queue_size"])
    crop_q = Queue(maxsize=args["queue_size"])
    pool = Pool(1, worker, (input_q,output_q,crop_q))
    pool2 = Pool(1, crop_worker, (crop_q,))
    # created a threaded video stream and start the FPS counter
    vs = WebcamVideoStream(src=args["input_device"]).start()
    fps = FPS().start()

    # Define the output codec and create VideoWriter object
    if args["output"]:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('outputs/{}.avi'.format(args["output_name"]),
                              fourcc, vs.getFPS()/args["num_workers"], (vs.getWidth(), vs.getHeight()))


    # Start reading and treating the video stream
    if args["display"] > 0:
        print()
        print("=====================================================================")
        print("Starting video acquisition. Press 'q' (on the video windows) to stop.")
        print("=====================================================================")
        print()

    countFrame = 0
    while True:
        # Capture frame-by-frame
        ret, frame = vs.read()
        countFrame = countFrame + 1
        if ret:
            input_q.put(frame)
            output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
            #print("size of crop_q:", crop_q.qsize())
            # write the frame
            if args["output"]:
                out.write(output_rgb)

            # Display the resulting frame
            if args["display"]:
                ## full screen
                if args["full_screen"]:
                    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("frame", output_rgb)

                fps.update()
            elif countFrame >= args["num_frames"]:
                break

        else:
            break

        k = cv2.waitKey(1)        
        if k == ord('q'):
            #path = "/home/mohak/Desktop/image/"
            #test = os.listdir(path)
            #for item in test:
                #if item.endswith(".jpg"):
                    #os.remove(os.path.join(path, item))
                #else:
                    #pass
            break
            cv2.destroyAllWindows()

    # When everything done, release the capture
    fps.stop()
    pool.terminate()
    pool2.terminate()
    vs.stop()
    if args["output"]:
        out.release()
    cv2.destroyAllWindows()
