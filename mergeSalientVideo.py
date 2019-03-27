import cv2
imgDir = '/Users/momo/Desktop/wy/findx2/'
videoName = 'findx2'
fps = 30  #TODO: get real fps from original video
im = cv2.imread(imgDir+videoName+"_s0.jpg")
output_size = (im.shape[1],im.shape[0])
greyWriter = cv2.VideoWriter(videoName + '_sal.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, output_size, 0)
colorWriter = cv2.VideoWriter(videoName + '_ori.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, output_size)
for i in xrange(215):
    greyname = imgDir+videoName+"_s"+str(i)+".jpg"
    grey = cv2.imread(greyname)
    print i,greyname, im.shape
    b,g,r = cv2.split(grey)
    greyWriter.write(b)
    colorname = imgDir+videoName+"_o"+str(i)+".jpg"
    color = cv2.imread(colorname)
    print i,greyname,colorname, grey.shape, color.shape
    colorWriter.write(color)
colorWriter.release()
greyWriter.release()
