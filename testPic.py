import cv2
from SaliconRes import Salicon
sal = Salicon()
map = sal.compute_saliency('face.jpg')
print map.shape
cv2.imwrite('face_out.jpg', map*255)
