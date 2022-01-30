from ASLT import ASLTranslator

with open('words.txt') as file:
    words = [line.rstrip() for line in file]
aslt = ASLTranslator('new_model.h5', words)

def StartVideo():
    aslt.RunVideo(interval = 30, threshold = 0.7, draw=[False, False, True, True])

def TrainModel():
    sequences, labels = aslt.GetDataFromPath('test_data')
    #create new model here

###################################################################

StartVideo()
#TrainModel()