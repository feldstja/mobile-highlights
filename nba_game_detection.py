import cv2
import numpy as np
import cgi, cgitb
import os
# import cvzone
cgitb.enable()


#
# from cvzone.ColorModule import ColorFinder
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

temp_ballys = cv2.imread('nba_logos/ballys_logo.png', 0)
sc_pct_ballys = 5300
thresh_ballys = .7


temp_espn = cv2.imread('nba_logos/espn_logo.png', 0)
sc_pct_espn = 5700
thresh_espn = .71

temp_tnt = cv2.imread('nba_logos/tnt_logo.png', 0)
sc_pct_tnt = 8000
thresh_tnt = .67

def detect_logo(img_gray, temp_logo, sc_pct, thresh):
    # print(sc_pct)
    # img_rgb = img.copy()
    # img_rgb =  img_rgb[800:1100, 950:1080]
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # template = cv2.imread('nba_logos/ballys_logo.png', 0)
    # template1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # img_ballys = img_full[arr]

    img_gray_logo = img_gray.copy()
    temp_logo_copy = temp_logo.copy()
    hw1 = temp_logo_copy.shape
    # print(hw1)

    scale_percent = sc_pct/hw1[0]
    width = int(temp_logo_copy.shape[1] * scale_percent / 100) #need to fix this so no /100
    height = int(temp_logo_copy.shape[0] * scale_percent / 100)
    dim = (width, height)
    # print(dim)

    # resize image
    resized = cv2.resize(temp_logo_copy, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(str(thresh), temp_logo)
    w, h = resized.shape[::-1]

    res = cv2.matchTemplate(img_gray_logo,resized,cv2.TM_CCOEFF_NORMED)
    # threshold = 0.58 #ballys
    # threshold = thresh
    # threshold = 0.7 #espn
    # print(res)
    loc = np.where(res >= thresh)
    # # print(loc)
    # # print(max(res[0]))
    i = 0
    for pt in zip(*loc[::-1]):
        # print(pt)
        i+= 1
        # if (i == 1):
        cv2.rectangle(img_gray_logo, pt, (pt[0] + w, pt[1] + h), (255,100,0), 2)

    cv2.imshow(str(sc_pct), img_gray_logo)
    return(len(loc[0]))

    # # print(i)
    # # return(img_gray_logo)
    # return(i)

    # return(img_rgb)
def detect_tv(img):
    # global temp_ballys
    img_full = img.copy()
    img_full = cv2.cvtColor(img_full, cv2.COLOR_RGB2GRAY)
    width = img_full.shape[1]
    height = img_full.shape[0]
    # print(width, height)

    img_ballys = img_full[int(.9*height):, int(.45*width):int(.56*width)]
    img_espn = img_full[int(.8 * height):, int(.8 * width):]
    img_tnt = img_full[int(.75 * height):int(.9 * height), int(.85 * width):int(.95 * width)]
    # _, img_espn = cv2.threshold(img_espn, 127, 255, cv2.THRESH_BINARY)

    ballys_i = detect_logo(img_ballys, temp_ballys, sc_pct_ballys, thresh_ballys)
    espn_i = detect_logo(img_espn, temp_espn, sc_pct_espn, thresh_espn)
    tnt_i = detect_logo(img_tnt, temp_tnt, sc_pct_tnt, thresh_tnt)
    # cv2.imshow(img_ballys_proc)
    if(ballys_i > 0):
        print("BALLYS: ", ballys_i)
    if(espn_i > 0):
        print("ESPN: ", espn_i)
    if(tnt_i > 0):
        print("TNT: ", tnt_i)
    return (img_tnt)

    # IDEA: add up i scores on video, first to 5000 or something like that is the network, or just highest one is
    # IDEA: once network is detected, watch just area of score board keep track of teams, scores, time, quarter, shotclock,
    #       when no score board is detected, all variables stay what they were before then can be updated later


# form = cgi.FieldStorage()
# @app.route("/index")
# def get_video():
# cgitb.enable()
# form = cgi.FieldStorage()
# # Get filename here.
# fileitem = form['file']
# # Test if the file was uploaded
# if fileitem.filename:
#    # strip leading path from file name to avoid
#    # directory traversal attacks
#    fn = os.path.basename(fileitem.filename)
#    open('/tmp/' + fn, 'wb').write(fileitem.file.read())
#    message = 'The file "' + fn + '" was uploaded successfully'
# else:
#    message = 'No file was uploaded'
# print """\
# Content-Type: text/html\n
# <html>
# <body>
#    <p>%s</p>
# </body>
# </html>
# """ % (message,)# cgitb.enable()

class upfile(object):

    def __init__(self):
        self.script_dir = os.path.dirname(__file__)
        self.errors = []


    def __call__(self, environ, start_response):


        f = open(os.path.join(self.script_dir, 'index.html'))
        self.output = f.read()
        f.close()

        self.response_content_type = 'text/html;charset=UTF-8'
        fields = None
        if 'POST' == environ['REQUEST_METHOD'] :
            fields = cgi.FieldStorage(fp=environ['wsgi.input'],environ=environ, keep_blank_values=1)
            fileitem = fields['file']
            fn = os.path.basename(fileitem.filename)
            open('uploads/' + fn, 'wb').write(fileitem.file.read())


        self.output = self.output % {"filepath":str(fields)} # Just to see the contents

        response_headers = [('Content-type', self.response_content_type),('Content-Length', str(len(self.output)))]
        status = '200 OK'
        start_response(status, response_headers)
        return [self.output]

application = upfile()

# form=cgi.FieldStorage()
# who = form.getvalue('who')	# we expect certain fields to be there, value may be None if field is left blank
# names = form.keys()		# all the input names for which values exist
# sports = form.getlist('sports') # which might be a <select multiple name="sports">


# cap = cv2.VideoCapture("nba_highlights/ESPN Nets Grizzlies.mp4")
# cap = cv2.VideoCapture("nba_highlights/ESPN Lakers Warriors.mp4")
# cap = cv2.VideoCapture("nba_highlights/Ballys Bucks Wizards.mp4")
# cap = cv2.VideoCapture("nba_highlights/Ballys Lakers Pelicans.mp4")
cap = cv2.VideoCapture("nba_highlights/TNT Clippers Nuggets.mp4")
# myColorFinder = ColorFinder()

while True:
    success, img = cap.read()
    # img = cv2.GaussianBlur(img, (9, 11), 0)
    # img = cv2.blur(img,(3,3))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    # cv2.imshow("img", img)
    # img1 = img[643:702, 800:]
    # img2 = cv2.imread('nba_logos/Screen Shot 2022-03-25 at 1.02.47 PM.png', 0)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    #
    # match = cv2.matchTemplate(gray, gray2, cv2.TM_CCOEFF_NORMED)
    # img2 = img.copy()
    # template = cv2.imread('nba_logos/Screen Shot 2022-03-25 at 1.02.47 PM.png',0)
    # w, h = template.shape[::-1]
    #
    # # All the 6 methods for comparison in a list
    # # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    # #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # res = cv2.matchTemplate(img2,template,cv2.TM_SQDIFF)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    # #     top_left = min_loc
    # # else:
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    #
    # cv2.rectangle(img2,top_left, bottom_right, 255, 2)
    # cv2.imshow("mat", img2)
    #
    # for meth in methods:
    #     img = img2.copy()
    #     method = eval(meth)
    #
    #     # Apply template Matching
    #     res = cv2.matchTemplate(img,template,method)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #
    #     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #         top_left = min_loc
    #     else:
    #         top_left = max_loc
    #     bottom_right = (top_left[0] + w, top_left[1] + h)
    #
    #     cv2.rectangle(img,top_left, bottom_right, 255, 2)
    #
    #     plt.subplot(121),plt.imshow(res,cmap = 'gray')
    #     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(122),plt.imshow(img,cmap = 'gray')
    #     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #     plt.suptitle(meth)
    #
    #     plt.show()
    #
    # cv2.rectangle(img, (700, 800),
    #               (1100, 1100),
    #               (0, 255, 0), 5)
    # cv2.rectangle(img, (750, 900),
    #               (1100, 1080),
    #               (255, 255, 0), 5)

    # img_rgb = img.copy()
    # img_rgb =  img_rgb[800:1100, 950:1080]
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # template = cv2.imread('nba_logos/ballys_logo.png', 0)
    # # template1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # hw1 = template.shape
    # # print(hw1)
    #
    # # scale_percent = 5700/hw1[0] # espn screenshot 1.02.47
    # scale_percent = 5300/hw1[0] # ballys_logo
    # width = int(template.shape[1] * scale_percent / 100)
    # height = int(template.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # print(dim)
    # # resize image
    # resized = cv2.resize(template, dim, interpolation = cv2.INTER_AREA)
    #
    # w, h = resized.shape[::-1]
    #
    # res = cv2.matchTemplate(img_gray,resized,cv2.TM_CCOEFF_NORMED)
    # threshold = 0.58 #ballys
    # # threshold = 0.7 #espn
    # # print(res)
    # loc = np.where(res >= threshold)
    # print(len(loc))
    # print(loc)
    # # print(max(res[0]))
    # i = 0
    # for pt in zip(*loc[::-1]):
    #     i+= 1
    #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,100,0), 2)
    #
    # # cv2.imwrite('res.png',img_rgb)
    # if ( i > 0):
    #     print(i)
    # cv2.imshow("mat", img_rgb)

    # processed_pic = img[800:, 950:1080]

    # cv2.imshow("img", img)
    # width = img.shape[1]
    # height = img.shape[0]
    # print(width, height)
    dim = (1920, 1080)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # width = resized_img.shape[1]
    # height = resized_img.shape[0]
    # print(width, height)

    processed_pic = detect_tv(resized_img)
    # cv2.imshow("mat", processed_pic)

    # ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
    #
    # contours,h = cv2.findContours(thresh,1,2)
    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    #     # print(len(approx))
    #     # if len(approx)==5:
    #     #     print ("pentagon")
    #     #     cv2.drawContours(img,[cnt],0,255,-1)
    #     # elif len(approx)==3:
    #     #     print ("triangle")
    #     #     cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    #
    #     # if(len(approx) == 2):
    #     #     # print("hello")
    #     #     cv2.drawContours( img,[cnt],0,(255,0,0),-1)
    #
    #     if len(approx)==4:
    #         width = abs(approx[0][0][0] - approx[2][0][0])
    #         height = abs(approx[0][0][1] - approx[2][0][1])
    #         if(width > 400 and width < 1000 and height > 20 and height < 200):
    #             print(approx)
    #             print('h: ', height, ' w: ', width)
    #             # print ("square")
    #             # print(approx)
    #             cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    #
    #


# h:  59  w:  417

# [[[ 34 643]]
#
#  [[592 641]]
#
#  [[593 702]]
#
#  [[ 35 703]]]
# h:  59  w:  559
# [[[ 34 643]]
#
#  [[592 641]]
#
#  [[593 701]]
#
#  [[ 35 703]]]
# h:  58  w:  559
        # elif len(approx) == 9:
        #     print ("half-circle")
        #     cv2.drawContours(img,[cnt],0,(255,255,0),-1)
        # elif len(approx) > 15:
        #     print ("circle")
        #     cv2.drawContours(img,[cnt],0,(0,255,255),-1)


    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # screenshot = cv2.imread("Screen Shot 2022-03-24 at 6.40.43 PM.png")
    # hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    # lower_orange =  np.array([10, 44, 45])
    # upper_orange = np.array([15, 61, 37])

    # boxes, weights = hog.detectMultiScale(img, winStride=(8,8) )
    # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    # for (xA, yA, xB, yB) in boxes:
    #     # display the detected boxes in the colour picture
    #     cv2.rectangle(img, (xA, yA), (xB, yB),
    #                       (0, 255, 0), 2)


    # mask = cv2.inRange(hsv, upper_orange, lower_orange)
    # cv2.imshow('game', hsv)
    # cv2.imshow('hello', mask)
    # result = cv2.bitwise_and(screenshot, screenshot, mask=mask)

    # img1 = img[643:702, 34:720]
    # hw = img1.shape
    # # print(hw)
    # cv2.imshow("scorebug", img1)
    # cv2.rectangle(img, (34, 643),
    #               (720, 702),
    #               (0, 255, 0), 5)
    # cv2.imshow("edges", edges)
    # cv2.imshow("test", img)

    # cv2.imshow('ball game', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
