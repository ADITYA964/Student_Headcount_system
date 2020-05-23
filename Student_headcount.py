
import mtcnn  
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN  


filename0 = 'test image.jpg'
pixels0=pyplot.imread(filename0)
detector=MTCNN() 
FACES0=detector.detect_faces(pixels0)   

for face0 in FACES0:
                                                                                                    # Printing the coordinates  and features or keypoints of each face in input image 'dif1.jpg' 
        print("Bounding box coordinates of each face"," ",face0['box'])
        
        print("Landmark coordinates of each face", " ", face0['keypoints'])
        print("\n")
  


def draw_faces(filename, result_list):
	
	data = pyplot.imread(filename)	                                                                          # We load the image  from filename																												# We load the image  from filename
	
	for i in range(len(result_list)):                                                                         # Plot each face as a subplot																												# plot each face as a subplot
		
		x1,y1, width, height = result_list[i]['box']                                                      # Get coordinates																				      
		
		x2, y2 = (x1 + width), (y1 + height)
		
		pyplot.subplot(1, len(result_list), i+1)	                                                  # Define subplot																						 define subplot
		pyplot.axis('off')
		
		pyplot.imshow(data[y1:y2, x1:x2])                                                                 # Plot f ace
	
	pyplot.show()                                                         

    
print("\nTotal number of student faces in the classroom  =  ",len(FACES0))
print("\n\nTweaking the faces of students to give it as output")

draw_faces(filename0, FACES0)



# Saving model to disk
pickle.dump(detector, open('model.pkl','wb'))


