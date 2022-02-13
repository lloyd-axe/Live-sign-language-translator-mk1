from ASLT import ASLTranslator
MODEL_PATH = '../model.h5'
WORDLIST_PATH = '../word_list.txt'

with open(WORDLIST_PATH) as file:
    words = [line.rstrip() for line in file]
print(words)

aslt = ASLTranslator(MODEL_PATH, words)

def Run():
    aslt.StartCapture(threshold = 0.8, draw = [True, False, True, True])
    
def TrainNewGestures(wordsList, data_path, sampleCount):
    for word in wordsList:
        aslt.CollectDataforWord(word, data_path, sampleCount = sampleCount)
    sequences, labels = aslt.GetNpyDataFromPath(data_path)
    
    #create new model
    model = aslt.CreateModel()
    model_name = 'new_model.h5'
    x_train, x_test, y_train, y_test = aslt.SplitData(sequences, labels)
    aslt.TrainModel(model, x_train, y_train, epochs = 100, model_name = model_name)
    
    
#TEST ------------------------------
Run()

#train model to learn new words(gestures)
#newWords = ['yes', 'no', 'good']
#TrainNewGestures(newWords,'Data', 10)
