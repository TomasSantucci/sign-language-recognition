import random
import requests
import cv2

def getWordRepo():
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
    response = requests.get(word_site)
    return response.content.decode('utf-8').splitlines()

def newWord(words):
    word = random.choice(words).upper()
    rem = word
    guessed = ""
    return word, rem, guessed

def drawSquare(img,letter,col,start_x,start_y,square_size):
    img = cv2.rectangle(img,(start_x, start_y), (start_x + square_size, start_y + square_size), (255,255,255), -1)
    img = cv2.putText(img, letter, (start_x + 10,start_y + square_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2, cv2.LINE_AA)
    return img