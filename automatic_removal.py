# Autmatic Detection

# selected blemish patch from mouse click
def selectedBlemish(x,y,r):
    global i
    crop_img = source[y:(y+2*r), x:(x+2*r)]     
    #i = i + 1
    #cv2.imwrite("blemish-"+ str(i) +".png",crop_img)
    return identifybestPatch(x,y,r)

# get the best gradient patch around the blemish
def identifybestPatch(x,y,r):
    #Nearby Patches in all 8 directions
    patches={}

    key1tup = appendDictionary(x+2*r,y)
    patches['Key1'] = (x+2*r,y,key1tup[0],key1tup[1])

    key2tup = appendDictionary(x+2*r,y+r)
    patches['Key2'] = (x+2*r,y+r,key2tup[0],key2tup[1])

    key3tup = appendDictionary(x-2*r,y)
    patches['Key3'] = (x-2*r,y,key3tup[0],key3tup[1])

    key4tup = appendDictionary(x-2*r,y-r)
    patches['Key4'] = (x-2*r,y-r,key4tup[0],key4tup[1])

    key5tup = appendDictionary(x,y+2*r)
    patches['Key5'] = (x,y+2*r,key5tup[0],key5tup[1])

    key6tup = appendDictionary(x+r,y+2*r)
    patches['Key6'] = (x+r,y+2*r,key6tup[0],key6tup[1])

    key7tup = appendDictionary(x,y-2*r)
    patches['Key7'] = (x,y-2*r,key7tup[0],key7tup[1])

    key8tup = appendDictionary(x-r,y-2*r)
    patches['Key8'] = (x-r,y-2*r,key8tup[0],key8tup[1])

    #print(patches)
    findlowx = {}
    findlowy = {}
    for key, (x, y, gx, gy) in patches.items():
        findlowx[key] = gx

    for key, (x, y, gx, gy) in patches.items():
        findlowy[key] = gy

    y_key_min = min(findlowy.keys(), key=(lambda k: findlowy[k]))
    x_key_min = min(findlowx.keys(), key=(lambda k: findlowx[k]))

    if x_key_min == y_key_min:
        return patches[x_key_min][0], patches[x_key_min][1]
    else:
        #print("Return x & y conflict, Can take help from FFT")
        return patches[x_key_min][0], patches[x_key_min][1]

# Get the gradients of x and y
def appendDictionary(x,y):
    crop_img = source[y:(y+2*r), x:(x+2*r)]    
    gradient_x, gradient_y = sobelfilter(crop_img)
    return gradient_x, gradient_y

# Apply sobel filter
def sobelfilter(crop_img):
    sobelx64f = cv2.Sobel(crop_img,cv2.CV_64F,1,0,ksize=3)
    abs_xsobel64f = np.absolute(sobelx64f)
    sobel_x8u = np.uint8(abs_xsobel64f)
    gradient_x = np.mean(sobel_x8u)

    sobely64f = cv2.Sobel(crop_img,cv2.CV_64F,0,1,ksize=3)
    abs_ysobel64f = np.absolute(sobely64f)
    sobel_y8u = np.uint8(abs_ysobel64f)
    gradient_y = np.mean(sobel_y8u)

    return gradient_x, gradient_y

# remove the blemish
def blemishRemoval(x, y):
  # Referencing global variables 
  global r, source
  # Mark the center
  blemishLocation = (x,y)
  # print(blemishLocation)    
  newX, newY = selectedBlemish(x,y,r)
  newPatch = source[newY:(newY+2*r), newX:(newX+2*r)]
  cv2.imwrite("newpatch.png",newPatch)
  # Create mask for the new Patch  
  mask = 255 * np.ones(newPatch.shape, newPatch.dtype) 
  source = cv2.seamlessClone(newPatch, source, mask, blemishLocation, cv2.NORMAL_CLONE) 
  cv2.imshow("Blemish Removal App",source)

# Detect coordinates based on filter values
def coordinateDetection(boolArea, valArea, boolCircularity, valCircularity, boolConvexity, valConvexity, boolInertia, valInertia):
    # Read image
    im = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200


    # Filter by Area.
    params.filterByArea = boolArea
    params.minArea = int(valArea)

    # Filter by Circularity
    params.filterByCircularity = boolCircularity
    params.minCircularity = float(valCircularity)

    # Filter by Convexity
    params.filterByConvexity = boolConvexity
    params.minConvexity = float(valConvexity)
        
    # Filter by Inertia
    params.filterByInertia = boolInertia
    params.minInertiaRatio = float(valInertia)

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)


    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    # print(str(int(keypoints[0].pt[0])))
    # print(str(int(keypoints[0].pt[1])))
    # (x, y) = keypoints[0].pt
    # print(x,y)

    # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    
    return keypoints

if __name__ == '__main__':
    import cv2
    import numpy as np

    # Lists to store the points
    r = 15
    i = 0

    # Get user input values
    imgPath = input("Enter the image name\n")
    valArea = 0.0
    valCircularity = 0.0
    valConvexity = 0.0
    valInertia = 0.0
    boolArea = False
    boolCircularity = False
    boolConvexity = False
    boolInertia = False

    userArea = input("Filter by Area? True or False\n")
    if userArea == "True":
        boolArea = True
        valArea = input("Enter area filter value: 0-1500\n")
    
    userCircularity = input("Filter by Circularity? True or False\n")
    if userCircularity == "True":
        boolCircularity = True
        valCircularity = input("Enter area filter value: 0.0-1.0\n")
    
    userConvexity = input("Filter by Convexity? True or False\n")
    if userConvexity == "True":
        boolConvexity = True
        valConvexity = input("Enter area filter value: 0.0-1.0\n")
    
    userInertia = input("Filter by Inertia? True or False\n")
    if userInertia == "True":
        boolInertia = True
        valInertia = input("Enter area filter value: 0.0-1.0\n")
    
    r = input("Select patch radius (recommended 10-30)\n")
    r = int(r)
    source = cv2.imread(imgPath,1)
    # Make a dummy image, will be useful to clear the drawing
    h, w, c = source.shape
    # print(h, w, c)
    cv2.namedWindow("Blemish Removal App")
    # highgui function called when mouse events occur
    print(f"Using a patch of radius {r}:")


    acne_points = coordinateDetection(boolArea, valArea, boolCircularity, valCircularity, boolConvexity, valConvexity, boolInertia, valInertia)
    for point in acne_points:
        x_cord = int(point.pt[0])
        # print(x_cord)
        y_cord = int(point.pt[1])
        if (x_cord > 40 and x_cord < w - 40 and y_cord > 40 and y_cord < h - 40):
            blemishRemoval(x_cord, y_cord)

    # Waits for escape to be pressed to exit
    k = 0
    while k!=27 :
        cv2.imshow("Blemish Removal App", source)
        k = cv2.waitKey(20) & 0xFF
    cv2.destroyAllWindows()