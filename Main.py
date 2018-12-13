import numpy as np
import pyautogui as gui
import pytesseract as ocr
import pyscreenshot
import cv2
import time
import imutils
import os
import aiml
from googletrans import Translator
import random

img = None
area = (0,0,365,405)
translator = Translator()

print('Initializing kernel')
kernel = aiml.Kernel()
# Create the kernel and learn AIML files
if os.path.isfile("res/brain.brn"):
    kernel.bootstrap(brainFile = "res/brain.brn")

def preview(imread):
    cv2.imshow('IMG',imread)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def colorContour(im, crange, minx=0, maxx=40, miny=0, maxy=40, preview=False):

        # BGR Format !!! ([min b,g,r], [max b,g,r])
        imread = im
        colorRange = [crange]

        # loop over the boundaries
        for (lower, upper) in colorRange:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(imread, lower, upper)
            output = cv2.cvtColor(cv2.bitwise_and(imread, imread, mask = mask), cv2.COLOR_RGB2GRAY)

            cnts = cv2.findContours(output.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            pv = imread

            # loop over the contours
            i = 0
            for c in cnts:

                # print(cv2.boundingRect(c)[2], cv2.boundingRect(c)[3])
                # if the contour is detected, return True
                if ((cv2.boundingRect(c)[2] >= minx) & (cv2.boundingRect(c)[2] <= maxx)) & ((cv2.boundingRect(c)[3] >= miny) & (cv2.boundingRect(c)[3] <= maxy)) & (preview is False):
                    return cnts

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                if preview:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(pv, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(pv, str(i),(x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,0,255),1,cv2.LINE_AA)
                    i += 1

            if preview:
                pv = cv2.resize(pv, None, fy=1, fx=1)
                output = cv2.resize(output, None, fy=1, fx=1)
                cv2.imshow("filter", pv)
                # cv2.imshow("filter", output)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
            i = 0

def newMessage():
    print('New msg detected')
    newMsgImg = img[75:403, 6:80]
    try:
        click = colorContour(newMsgImg, ([90,190,10],[100,198,15]), minx=16, miny=16, maxx=20, maxy=20, preview=False)
        clickx, clicky = cv2.boundingRect(click[0])[0], cv2.boundingRect(click[0])[1]
        gui.click(6+clickx,75+clicky)
    except TypeError:
        return

lastMsg = None
def getMessageText():
    print('Getting text')
    global lastMsg
    msgArea = img[78:346, 84:365]
    try:
        msgBox = colorContour(msgArea, ([242,236,235],[245,239,237]), minx=15, miny=8, maxx=1000, maxy=1000, preview=False)
        x, y, w, h = cv2.boundingRect(msgBox[0])
        msgArea = cv2.cvtColor(msgArea, cv2.COLOR_RGB2GRAY)
        msgArea = cv2.resize(msgArea[y-5:y+h+5,x-5:x+w+5], None, fx=3, fy=3, interpolation=cv2.INTER_BITS)

        msg = ocr.image_to_string(msgArea)
        if msg == lastMsg:
            return
        lastMsg = msg
        print('Text:', ocr.image_to_string(msgArea))

        msg = textPreprocess(msg)
        reply_msg(msg)
    except TypeError as e:
        print(e)
        return

def textPreprocess(text):
    print('Preprocess Text')
    log = open('res/log.txt', 'a')
    text = text.replace('\n', '').lower()

    # Filters the input text
    with open('res/inputfilter.txt', 'r') as f:
        filters = f.read()

        for line in filters.split('\n'):
            ori = line.split('|')[0]

            for word in line.split('|'):
                if word in text:
                    text = text.replace(word, ori)
                    break

    # Translates the input text to English, and output to Indonesia
    print('Translating Text')
    try:
        text = translator.translate(text, src='id', dest='en')
        respond = kernel.respond(text.text)
        respond = translator.translate(respond, src='en', dest='id')
        log.write('\n\n%s --> %s' % (text.origin, text.text))
        log.write('\n%s --> %s' % (respond.origin, respond.text))
    except Exception as e:
        print(e)
        return

    respond = respond.text.lower()

    with open('res/outputfilter.txt', 'r') as f:
        filters = f.read()

        for line in filters.split('\n'):
            ori = line.split('|')[0]

            for word in line.split('|'):
                if ori in respond:
                    respond = respond.replace(ori, line.split('|')[random.randrange(1, len(line.split('|'))-1)])

    if 'alice' in respond:
        respond = respond.replace('alice', 'A.N.S.O.S')
    if 'bot' in respond:
        respond = respond.replace('bot', 'servant')

    log.write('\n%s' % respond)
    log.close()

    return respond

def reply_msg(msg):
    gui.click(160,376)
    gui.typewrite(msg, interval=0.01)
    gui.press('enter')

while True:
    print('Taking screenshot')
    img = cv2.cvtColor(np.array(pyscreenshot.grab(bbox=area)), cv2.COLOR_BGR2RGB)
    # preview(img)

    # newMessage()
    getMessageText()

    time.sleep(5)

    gui.moveRel(3,0)
    gui.moveRel(-3,0)