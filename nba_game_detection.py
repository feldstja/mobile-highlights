import cv2
import numpy as np

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
    return (res) # return the best matched network

    # IDEA: add up i scores on video, first to 5000 or something like that is the network, or just highest one is
    # IDEA: once network is detected, watch just area of score board keep track of teams, scores, time, quarter, shotclock,
    #       when no score board is detected, all variables stay what they were before then can be updated later

# These are the video options we have so far
cap = cv2.VideoCapture("nba_highlights/ESPN Nets Grizzlies.mp4")
# cap = cv2.VideoCapture("nba_highlights/ESPN Lakers Warriors.mp4")
# cap = cv2.VideoCapture("nba_highlights/Ballys Bucks Wizards.mp4")
# cap = cv2.VideoCapture("nba_highlights/Ballys Lakers Pelicans.mp4")
# cap = cv2.VideoCapture("nba_highlights/TNT Clippers Nuggets.mp4")

# Here we get the video, resize it to conform to normal tv size,
# find out which network its on, isolate score bug, display it in the middle,
# and crop it to make it all instagram compatible
while True:
    success, img = cap.read()

    dim = (1920, 1080)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    score_bug =[]

    # Find network here
    network, score = detect_tv(img)
    # w, h = 0, 0
    # width, height = dim

    # if score is high enough, isolate scorebug and blackout original score bug
    if(score > 0):
        if network == "ballys":
            score_bug = img[980:1040, 40:960].copy()
            cv2.rectangle(img, (40,980), (960,1040), (0,0,0), -1) #BALLYS SCOREBUG
        elif network == "espn":
            score_bug = img[960:1060, 50:1080].copy()
            cv2.rectangle(img, (50,960), (1080,1060), (0,0,0), -1) #ESPN SCOREBUG
        elif network == "tnt":
            score_bug = img[860:950, 1280:1680].copy()
            cv2.rectangle(img, (1250,850), (1700,960), (0,0,0), -1) #TNT SCOREBUG
        # print(network)

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
