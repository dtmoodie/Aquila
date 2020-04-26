import aquila

graph = aquila.Graph()

devices = aquila.framegrabbers.listDataSources()
fg = aquila.framegrabbers.frame_grabber_openni2()
graph.addNode(fg)
fg.loadData(devices[0])
disp = aquila.nodes.OGLImageDisplay(name='depth', image=(fg, fg.getOutput('depth')))

cam = aquila.framegrabbers.FrameGrabber()
graph.addNode(cam)

cam.loadData('0')


foreground = aquila.nodes.ConvolutionL2ForegroundEstimate(input=(fg, fg.getOutput('xyz')), kernel_size=9, distance_threshold=10000)
foregrounddisp = aquila.nodes.OGLImageDisplay(image=(foreground, foreground.getOutput('distance')))
face = aquila.nodes.HaarFaceDetector(model_file='/home/asiansensation/code/opencv/data/haarcascades/haarcascade_frontalface_default.xml', input=cam)
disp = aquila.nodes.DrawDetections(image=cam, detections=face)
img_disp = aquila.nodes.OGLImageDisplay(name='image', image=disp)
graph.start()

fg.getOutputs()[0].data
