import cv2
import numpy as np
from imutils import paths  
import os 
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=(".tif", ".jpg", ".jpeg", ".png", ".bmp"), contains=contains)

def list_files(basePath, validExts=(".tif", ".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath


classes = {1: 'English',
           2: 'Malayalam',
           3: 'Hindi',
           4: 'Urdu',
           5: 'Bengali',
           6: 'Tamil',
           7: 'Telugu',
           8: 'Punjabi',
           9: 'Gujarati',
           10:'Oriya'} 

labels = []
dictionarySize = 60#60

BOW = cv2.BOWKMeansTrainer(dictionarySize)
sift = cv2.xfeatures2d.SIFT_create()

imagePaths = list(list_images("train_data")) #list(paths.list_images("train"))
#print(imagePaths)

for image in imagePaths:
	print("Processing image - ", image)
	label = image.split('/')[1]
	if label == 'English':
		labels.append(1)
	elif label == 'Malayalam':
		labels.append(2)
	elif label == 'Hindi':
		labels.append(3)
	elif label == 'Urdu':
		labels.append(4)
	elif label =='Bengali':
		labels.append(5)
	elif label =='Tamil':
		labels.append(6)
	elif label =='Telugu':
		labels.append(7)
	elif label =='Punjabi':
		labels.append(8)
	elif label =='Gujarati':
		labels.append(9)
	elif label =='Oriya':
		labels.append(10)
	elif label == 'English2':
		labels.append(1)

	image = image.replace("\\","")
	img = cv2.imread(image,0)
	kp, des = sift.detectAndCompute(img,None)
	BOW.add(des)

print("Creating Bag of Words")
dictionary = BOW.cluster()
print("Created Bag of Words")

sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
print("Setting vocabulary")
bowDiction.setVocabulary(dictionary)

desc = []
for image in imagePaths:
	print("Classifying image - ", image)
	image = image.replace("\\","")
	img = cv2.imread(image,0)	
	desc.extend(bowDiction.compute(img, sift.detect(img)))

print("Creating KNN model")
desc = np.array(desc).astype(np.float32)
labels = np.array(labels).astype(np.float32) 
knn = cv2.ml.KNearest_create()
knn.train(desc, cv2.ml.ROW_SAMPLE, labels)
print("Created KNN model")

expected_label = []
output_label = [] 
testPath = list(list_images("test_data")) #list(paths.list_images("test"))
print('Actual Class','\t-\t','Predicted Class')

for image in testPath:
	print("Processing test image - ", image)
	label = image.split('/')[1]
	image = image.replace("\\","")
	#img = cv2.imread(image,0)


	imgInput = cv2.imread(image)            
	imgGray = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)         
	# invert black and white
	newRet, binaryThreshold = cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY_INV)

	rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) #(15,10)
	rectdilation = cv2.dilate(binaryThreshold, rectkernel, iterations = 1)
	outputImage = imgInput.copy()
	img, npaContours, npaHierarchy = cv2.findContours(rectdilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)           

	MIN_CONTOUR_AREA = 1000
	lang = {'English': 0, 'Malayalam': 0, 'Hindi': 0, 'Urdu': 0, 'Bengali': 0, 'Tamil': 0, 'Telugu': 0, 'Punjabi': 0, 'Gujarati': 0, 'Oriya': 0}

	for npaContour in npaContours:                         
	    if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
	    	[intX, intY, intW, intH] = cv2.boundingRect(npaContour)
	    	cropped = binaryThreshold[intY:intY+intH, intX:intX+intW]
	    	try:
	    		feature = bowDiction.compute(cropped, sift.detect(cropped))
	    		feature = np.array(feature).astype(np.float32)
	    		ret, result, neighbour, distance = knn.findNearest(feature, 3)
	    		lang[classes[result[0][0]]] = lang[classes[result[0][0]]] + 1
	    	except:
	    		print('Exception raised on ', image)

	expected_label.append(label)
	output_lbl = max(lang, key=lang.get)
	output_label.append(output_lbl)
	print(label, "\t\t-\t", output_lbl)

print("Length of expected_label = ", len(expected_label))
print("\nAccuracy - ",accuracy_score(expected_label, output_label)*100, "%")
