from ASLT import ASLT
from keras.models import load_model

test = ASLT()

model = load_model('GIT//model.h5')

actions = ['hello', 'yes', 'no', 'thanks', 'iloveyou', 'xoxo']

test.run(model, actions, [False, False, False], threshold=0.85)

