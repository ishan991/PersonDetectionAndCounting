

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os

# import the necessary packages
import numpy as np
import imutils
import argparse
import time
import cv2
import os
import PySimpleGUI as sg

        
def open_img():
                

        # load the COCO class labels our YOLO model was trained on
        labelsPath = r"C:\Users\Hp\Desktop\yolo-object-detection\yolo-object-detection\yolo-coco\coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = r"C:\Users\Hp\Desktop\yolo-object-detection\yolo-object-detection\yolo-coco\yolov3.weights"
        configPath = r"C:\Users\Hp\Desktop\yolo-object-detection\yolo-object-detection\yolo-coco\yolov3.cfg"

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # load our input image and grab its spatial dimensions
        image = cv2.imread(filedialog.askopenfilename(title='open'))
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability) of
                        # the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > 0.3:
                                # scale the bounding box coordinates back relative to the
                                # size of the image, keeping in mind that YOLO actually
                                # returns the center (x, y)-coordinates of the bounding
                                # box followed by the boxes' width and height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top and
                                # and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates, confidences,
                                # and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3,0.3 )

        # count the number of persons
        count = 0

        # ensure at least one detection exists
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        
                        if(  LABELS[classIDs[i]])=='person':
                                count=count+1
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                                
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, color, 2)
        
        
                        
                                
        
        global bottomframe
        bottomframe = Frame(root,width=800,height=400, bg='gray78')
        bottomframe.pack( side = BOTTOM )


        img = cv2.resize(image, (400,400))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        panel = Label(bottomframe, image=img)
        panel.image = img
        panel.pack(ipadx=2,     ipady=2,        padx=10,        pady=10)

        
        
        panel2 = Label(bottomframe,
                fg = "midnight blue",
                bg = "gray78",
                font = "Helvetica 16 bold italic",
                text="Number of Persons detected")
        panel2.pack(ipadx=5,    ipady=5,        padx=5, pady=5)
        panel3=Label(bottomframe, 
                fg = "midnight blue",
                bg = "gray78",
                font = "Helvetica 16 bold italic",
                text=count)
        panel3.pack(ipadx=5,    ipady=5,        padx=5, pady=5)

def open_video():
        i_vid = r'videos\car_chase_01.mp4'
        # o_vid = r'videos\car_chase_01_out.mp4'
        y_path = r'yolo-coco'
        layout = 	[
                        [sg.Text('YOLO Video Player', size=(18,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
                        [sg.Text('Path to input video'), sg.In(i_vid,size=(40,1), key='input'), sg.FileBrowse()],
                        # [sg.Text('Path to output video'), sg.In(o_vid,size=(40,1), key='output'), sg.FileSaveAs()],
                        [sg.Text('Yolo base path'), sg.In(y_path,size=(40,1), key='yolo'), sg.FolderBrowse()],
                        [sg.Text('Confidence'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=.5, size=(15,15), key='confidence')],
                        [sg.Text('Threshold'), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.3, size=(15,15), key='threshold')],
                        [sg.OK(), sg.Cancel()]
                                ]

        win = sg.Window('YOLO Video',
				default_element_size=(14,1),
				text_justification='right',
				auto_size_text=False).Layout(layout)
        event, values = win.Read()
        if event is None or event =='Cancel':
                exit()
        args = values

        win.Close()


        # imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto

        # load the COCO class labels our YOLO model was trained on
        labelsPath = r"C:\Users\Hp\Desktop\yolo-object-detection\yolo-object-detection\yolo-coco\coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = r"C:\Users\Hp\Desktop\yolo-object-detection\yolo-object-detection\yolo-coco\yolov3.weights"
        configPath = r"C:\Users\Hp\Desktop\yolo-object-detection\yolo-object-detection\yolo-coco\yolov3.cfg"

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        vs = cv2.VideoCapture(args["input"])
        writer = None
        (W, H) = (None, None)

        # try to determine the total number of frames in the video file
        try:
                prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                       else cv2.CAP_PROP_FRAME_COUNT
                total = int(vs.get(prop))
                print("[INFO] {} total frames in video".format(total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
                print("[INFO] could not determine # of frames in video")
                print("[INFO] no approx. completion time can be provided")
                total = -1

        # loop over frames from the video file stream
        win_started = False
        while True:
                # read the next frame from the file
                (grabbed, frame) = vs.read()

                # if the frame was not grabbed, then we have reached the end
                # of the stream
                if not grabbed:
                        break

                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                        (H, W) = frame.shape[:2]

                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                             swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()

                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                        # loop over each of the detections
                        for detection in output:
                                # extract the class ID and confidence (i.e., probability)
                                # of the current object detection
                                scores = detection[5:]
                                classID = np.argmax(scores)
                                confidence = scores[classID]

                                # filter out weak predictions by ensuring the detected
                                # probability is greater than the minimum probability
                                if confidence > args["confidence"]:
                                        # scale the bounding box coordinates back relative to
                                        # the size of the image, keeping in mind that YOLO
                                        # actually returns the center (x, y)-coordinates of
                                        # the bounding box followed by the boxes' width and
                                        # height
                                        box = detection[0:4] * np.array([W, H, W, H])
                                        (centerX, centerY, width, height) = box.astype("int")

                                        # use the center (x, y)-coordinates to derive the top
                                        # and and left corner of the bounding box
                                        x = int(centerX - (width / 2))
                                        y = int(centerY - (height / 2))

                                        # update our list of bounding box coordinates,
                                        # confidences, and class IDs
                                        boxes.append([x, y, int(width), int(height)])
                                        confidences.append(float(confidence))
                                        classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                        args["threshold"])

                # ensure at least one detection exists
                if len(idxs) > 0:
                        # loop over the indexes we are keeping
                        for i in idxs.flatten():
                                # extract the bounding box coordinates
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (w, h) = (boxes[i][2], boxes[i][3])

                                # draw a bounding box rectangle and label on the frame
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                                           confidences[i])
                                cv2.putText(frame, text, (x, y - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # check if the video writer is None
                # if writer is None:
                # 	# initialize our video writer
                # 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                # 	writer = cv2.VideoWriter(args["output"], fourcc, 30,
                # 		(frame.shape[1], frame.shape[0]), True)
            #
                # 	# some information on processing single frame
                # 	if total > 0:
                # 		elap = (end - start)
                # 		print("[INFO] single frame took {:.4f} seconds".format(elap))
                # 		print("[INFO] estimated total time to finish: {:.4f}".format(
                # 			elap * total))

                # write the output frame to disk
                # writer.write(frame)
                imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto

                if not win_started:
                        win_started = True
                        layout = [
                                [sg.Text('Yolo Output')],
                                [sg.Image(data=imgbytes, key='_IMAGE_')],
                                [sg.Exit()]
                        ]
                        win = sg.Window('YOLO Output',
						default_element_size=(14, 1),
						text_justification='right',
						auto_size_text=False).Layout(layout).Finalize()
                        image_elem = win.FindElement('_IMAGE_')
                else:
                        image_elem.Update(data=imgbytes)

                event, values = win.Read(timeout=0)
                if event is None or event == 'Exit':
                        break


        win.Close()

        # release the file pointers
        print("[INFO] cleaning up...")
        writer.release()
        vs.release()
        
        
def exit():
        import sys; 
        sys.exit()

def refresh():
        bottomframe.destroy()
        
        
root = Tk()
root.title("Human Search and Rescue Application")
root.geometry("800x600")
root.resizable(width=True, height=True)
root.config(bg='gray78')




lbl = Label(root, 
                text=" Search and Rescue Application",
                compound = CENTER,
                padx = 10,
                fg = "midnight blue",
                bg = "gainsboro",
                font = "Georgia 30 bold "
                ).pack(ipadx=10,        ipady=10,       padx=30,        pady=10, fill = X)


frame = Frame(root, width=200,height=200, bg='gray78')
frame.pack()


btn = Button(frame, text='Open Image',
                fg = "midnight blue",
                bg = "lavender",
                font = "Helvetica 16 bold",
                command=open_img).pack(
                side=LEFT,
                ipadx=2,
                ipady=2,
                padx=10,
                pady=10)
btn = Button(frame, text='Open video',
                fg = "midnight blue",
                bg = "lavender",
                font = "Helvetica 16 bold",
                command=open_video).pack(
                side=LEFT,
                ipadx=2,
                ipady=2,
                padx=10,
                pady=10)
        
refresh_btn = Button(frame, text='Refresh',
                fg = "green",
                bg = "lavender",
                font = "Helvetica 16 bold",
                command=refresh).pack(
                side=LEFT,
                ipadx=2,
                ipady=2,
                padx=10,
                pady=10)        
exit_btn = Button(frame, text='Exit',
                fg = "red",
                bg = "lavender",
                font = "Helvetica 16 bold",
                command=exit).pack(
                side=RIGHT,
                ipadx=2,
                ipady=2,
                padx=10,
                pady=10)
                

bottomframe = Frame(root,width=800,height=400, bg='gray78')
                
root.mainloop()
