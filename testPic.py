import cv2
from SaliconMouth import Salicon
sal = Salicon()
map = sal.compute_saliency('face.jpg')
print map.shape
cv2.imwrite('mouth_out.jpg', map*255)
