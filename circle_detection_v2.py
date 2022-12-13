# Circle detection based on contours

import cv2
path = "grayscale.png"
gray = cv2.imread(path, 0)

# threshold
th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
# findcontours
cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

# filter by area
s1 = 3
s2 = 20
xcnts = []
g = 0
for cnt in cnts:
    
    if s1<cv2.contourArea(cnt) <s2:
        xcnts.append(cnt)
        print("amount:" + str(g))
        g = g + 1
        #print(cnt)
        i = 0
        for i in range(2):
            # print("(" + str(cnt[i][0][0]) + ", " + str(cnt[i][0][1]) + ")")
            # print(type(int(cnt[i][0][1])))
            cv2.circle(gray, ((cnt[i][0][0]), cnt[i][0][1]), 10, (150, 0, 150), 3)
#print(cnt)
cv2.imshow('output', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("\nDots number: {}".format(len(xcnts)))

