import cv2
import numpy as np
from PIL import Image
from pytesseract import pytesseract
# from py import image_to_string

# from pytesseract import *
# pytesseract.tesseract_cmd = r'/Users/jacobfeldstein/Downloads/mobile-highlight/pytesseract/'
# pytesseract.pytesseract.tesseract_cmd = r'C:/Users/jacobfeldstein/Library/Python/2.7/lib/python/site-packages'
ballys_score = tnt_score = espn_score = 0


#Global Variables so we only have to read the logos once
temp_ballys = cv2.imread('nba_logos/ballys_logo.png', 0)
sc_pct_ballys = 53
thresh_ballys = .7

temp_espn = cv2.imread('nba_logos/espn_logo.png', 0)
sc_pct_espn = 57
thresh_espn = .71

temp_tnt = cv2.imread('nba_logos/tnt_logo.png', 0)
sc_pct_tnt = 80
thresh_tnt = .67

# This function searches for the network logo in certain sections of the frame
def detect_logo(img_gray, temp_logo, sc_pct, thresh):
    img_gray_logo = img_gray.copy()
    temp_logo_copy = temp_logo.copy()
    hw1 = temp_logo_copy.shape

    # resizing image
    scale_percent = sc_pct/hw1[0]
    width = int(hw1[1] * scale_percent)
    height = int(hw1[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(temp_logo_copy, dim, interpolation = cv2.INTER_AREA)

    # w, h = resized.shape[::-1]
    # This is where we match the template to the image
    res = cv2.matchTemplate(img_gray_logo,resized,cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= thresh) #This is to make sure the match is good enough

    return(len(loc[0])) #If the logo is detected, it will return a number > 0


# This function splits up the frame into sections to feed to the logo detector
# and then finds out which network is it most likely on.
def detect_tv(img):
    global ballys_score, tnt_score, espn_score
    img_full = img.copy()
    img_full = cv2.cvtColor(img_full, cv2.COLOR_RGB2GRAY)
    width = img_full.shape[1]
    height = img_full.shape[0]

    # split up frame into relevent sections here
    img_ballys = img_full[int(.9*height):, int(.45*width):int(.56*width)]
    img_espn = img_full[int(.8 * height):, int(.8 * width):]
    img_tnt = img_full[int(.75 * height):int(.9 * height), int(.85 * width):int(.95 * width)]

    # feed relevent sections to logo detector and keep track of scores
    ballys_score += detect_logo(img_ballys, temp_ballys, sc_pct_ballys, thresh_ballys)
    espn_score += detect_logo(img_espn, temp_espn, sc_pct_espn, thresh_espn)
    tnt_score +=  detect_logo(img_tnt, temp_tnt, sc_pct_tnt, thresh_tnt)

    scores = [("ballys", ballys_score), ("espn", espn_score), ("tnt", tnt_score)]
    res = max(scores, key = lambda i : i[1])
    # print(res[0])
    return (res) # return the best matched network

    # IDEA: add up i scores on video, first to 5000 or something like that is the network, or just highest one is
    # IDEA: once network is detected, watch just area of score board keep track of teams, scores, time, quarter, shotclock,
    #       when no score board is detected, all variables stay what they were before then can be updated later

# These are the video options we have so far
# cap = cv2.VideoCapture("nba_highlights/ESPN Nets Grizzlies.mp4")
cap = cv2.VideoCapture("nba_highlights/ESPN Lakers Warriors.mp4")
# cap = cv2.VideoCapture("nba_highlights/Ballys Bucks Wizards.mp4")
# cap = cv2.VideoCapture("nba_highlights/Ballys Lakers Pelicans.mp4")
# cap = cv2.VideoCapture("nba_highlights/TNT Clippers Nuggets.mp4")

# Here we get the video, resize it to conform to normal tv size,
# find out which network its on, isolate score bug, display it in the middle,
# and crop it to make it all instagram compatible
score = 0
network = ''
count_test = 0
while True:
    success, img = cap.read()

    dim = (1920, 1080)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    score_bug =[]

    # Find network here

    if score < 5000:
        # print(score)
        network, score_i = detect_tv(img)
        score = score + score_i
    # w, h = 0, 0
    # width, height = dim

    # if score is high enough, isolate scorebug and blackout original score bug
    if(score > 0):
        if network == "ballys":
            score_bug = img[980:1040, 40:960].copy()
            cv2.rectangle(img, (40,980), (960,1040), (0,0,0), -1) #BALLYS SCOREBUG
        elif network == "espn":
            # print(score)
            count_test += 1

            score_bug = img[960:1060, 50:1080].copy()
            cv2.rectangle(img, (50,960), (1080,1060), (0,0,0), -1) #ESPN SCOREBUG
            # dimensions = score_bug.shape
            # score_bug = cv2.cvtColor(score_bug, cv2.COLOR_BGR2RGB)

            hImg, wImg, _ = score_bug.shape
            # cv2.rectangle(score_bug, (110,15), (225,70), (255,0,0), 1) #ESPN SCOREBUG
            # away_team = score_bug[15:70, 110:225].copy()
            # home_team = score_bug[15:70, 620:730].copy()
            # time_box = score_bug[10:60, 860:1000].copy()
            away_score = score_bug[10:70, 200:425].copy()
            home_score = score_bug[10:70, 430:600].copy()

            # gray_away = cv2.cvtColor(away_team, cv2.COLOR_BGR2GRAY)
            # gray_home = cv2.cvtColor(home_team, cv2.COLOR_BGR2GRAY)
            # gray_time = cv2.cvtColor(time_box, cv2.COLOR_BGR2GRAY)
            gray_as = cv2.cvtColor(away_score, cv2.COLOR_BGR2GRAY)
            gray_hs = cv2.cvtColor(home_score, cv2.COLOR_BGR2GRAY)
            cv2.imshow("homeg", gray_hs)
            cv2.imshow("awayg", gray_as)

            # ret,away_team = cv2.threshold(gray_away,150,255,cv2.THRESH_BINARY_INV)
            # ret2,home_team = cv2.threshold(gray_home,150,255,cv2.THRESH_BINARY_INV)
            # ret3,time_box = cv2.threshold(gray_time,150,255,cv2.THRESH_BINARY_INV)
            # gray_as = cv2.medianBlur(gray_as,5)
            # gray_hs = cv2.medianBlur(gray_hs,5)
            ret4,away_s = cv2.threshold(gray_as,210,255,cv2.THRESH_BINARY_INV)
            ret5,home_s = cv2.threshold(gray_hs,210,255,cv2.THRESH_BINARY_INV)
            kernel = np.ones((7,7),np.uint8)
            home_s = cv2.morphologyEx(home_s, cv2.MORPH_OPEN, kernel)

            # home_s = cv2.Canny(home_s, 100, 200)
            # coords = np.column_stack(np.where(home_s > 0))
            # angle = cv2.minAreaRect(coords)[-1]
            # if angle < -45:
            #     angle = -(90 + angle)
            # else:
            #     angle = -angle
            # (h, w) = home_s.shape[:2]
            # center = (w // 2, h // 2)
            # M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # home_s = cv2.warpAffine(home_s, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            cv2.imshow("away", away_s)
            cv2.imshow("home", home_s)

            # cv2.imshow("time", away_team)

            # print(pytesseract.image_to_string(away_team, config='tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip())
            # print(pytesseract.image_to_string(home_team, config='tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip())
            # print(pytesseract.image_to_string(time_box).strip())
            # print("away score: ", pytesseract.image_to_string(away_s, config='--psm 10'))
            if(count_test %20 == 3):
                print("away score: ", pytesseract.image_to_string(away_s, config='--psm 7 -c tessedit_char_whitelist=1234567890').strip())
            if(count_test %10 == 0):
                print("home score: ", pytesseract.image_to_string(home_s, config='--psm 7 -c tessedit_char_blacklist=O').strip())
            # print(pytesseract.image_to_string(home_s))
            # -c tessedit_char_whitelist=1234567890
            # score_bug2 = cv2.cvtColor(score_bug, cv2.COLOR_BGR2GRAY)
            # ret9,score_bug2 = cv2.threshold(score_bug2,150,255,cv2.THRESH_BINARY_INV)
            # boxes = pytesseract.image_to_boxes(score_bug2)
            # for b in boxes.splitlines():
            #     b = b.split(' ')
            #     print(b)
            #     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            #     cv2.rectangle(score_bug, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
            #     cv2.putText(score_bug, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)
            # print(pytesseract.image_to_string(score_bug2,config='--psm 1 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:').strip())
            # print("\nHELLO")
            # height = score_bug.shape[0]
            # width = score_bug.shape[1]
            # print("height: ", height)
            # print("width: ", width)
        elif network == "tnt":
            score_bug = img[860:950, 1280:1680].copy()
            cv2.rectangle(img, (1250,850), (1700,960), (0,0,0), -1) #TNT SCOREBUG
        # cv2.imshow("away team", away_team)
        # cv2.imshow("home team", home_team)

        # print(network)
        # custom_config = r'-l eng --oem 3 --psm 6'

    # If the score bug is found, move it to the center of the image
    x_offset = y_offset = 0
    if(len(score_bug)):
        x_offset=960-int(.5*(score_bug.shape[1]))
        y_offset=900
        img[y_offset:y_offset+score_bug.shape[0], x_offset:x_offset+score_bug.shape[1]] = score_bug

    # Show the instagram compatible part of the image
    cv2.imshow("img", img[:, 420:1500])
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
