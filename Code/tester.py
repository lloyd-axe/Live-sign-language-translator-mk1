from ASLT import ASLTranslator

with open('word_list.txt') as file:
    words = [line.rstrip() for line in file]

aslt = ASLTranslator('model.h5', words)
aslt.StartCapture(threshold = 0.8)
