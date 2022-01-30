from ASLT import ASLTranslator

with open('words.txt') as file:
    words = [line.rstrip() for line in file]
aslt = ASLTranslator('model.h5', words)
aslt.RunVideo(threshold = 0.7, draw=[False, False, True, True])
