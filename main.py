import cv2
import cvzone
from ultralytics import YOLO
import math
from utils import *
import numpy as np

wordRepo = getWordRepo()

model = YOLO("signlanguagen.pt")

classNames = model.names

height, width = 480, 640  # Define the size of the image
cap = cv2.VideoCapture(0)

# State variables
stats = False
word, rem, guessed = newWord(wordRepo)
guessedWords = []

while True:
    if rem == "": # Word is guessed
        guessedWords += [word]
        word, rem, guessed = newWord(wordRepo)

    success, img = cap.read()

    if not success:
        break

    results = model(img,stream=True)

    # -- Draw UI
    
    # Size of the square
    square_size = 40
    
    cv2.putText(img, f'COUNTER: {len(guessedWords)}', (20,460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (65,20,18), 2, cv2.LINE_AA)

    # Calculate the top-left corner of the square
    start_x = 20
    start_y = 20

    for letter in guessed:
        img = drawSquare(img,letter,(50,204,83),start_x,start_y,square_size)
        start_x += square_size + 2
    
    for letter in rem:
        img = drawSquare(img,letter,(27,27,207),start_x,start_y,square_size)
        start_x += square_size + 2
    
    # -- Main loop

    for r in results:
        boxes = r.boxes
        if rem == "":
            break
        
        for box in boxes:
            if rem == "":
                break

            conf = math.ceil((box.conf[0] * 100)) / 100

            if conf < 0.4:
                break

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
            
            clsname = classNames[int(box.cls[0])]

            if clsname == rem[0]:
                guessed += rem[0]
                rem = rem[1:]

            cvzone.putTextRect(img, f'{clsname} {conf}', (max(0,x1), max(35,y1)), scale = 0.8, thickness = 1)

    cv2.imshow("Image",img) # Show the frame

    # -- Keyboard events

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # Esc to exit
        break
    elif key == ord(' ') and rem != []: # Space to skip letter
        guessed += rem[0]
        rem = rem[1:]
    elif key == ord('s'): # s to show stats and exit
        stats = True
        break


if stats: # Display the stats screen
    start_x = 20
    start_y = 50

    img = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8)

    img = cv2.putText(img, f'GUESSED WORDS: ({len(guessedWords)})', (start_x,start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (85,40,38), 2, cv2.LINE_AA)

    for guessedWord in guessedWords:
        start_y += 40
        img = cv2.putText(img, guessedWord, (start_x,start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (85,40,38), 2, cv2.LINE_AA)

    cv2.imshow('Image', img)

    cv2.waitKey(0) # Press any key to exit

cv2.destroyAllWindows()
