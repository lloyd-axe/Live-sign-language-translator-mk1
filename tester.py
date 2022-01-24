from ASLT import ASLT
from keras.models import load_model

test = ASLT()

model = load_model('test_model.h5')

actions = ['hello', 'thanks', 'i love you']

test.run(model, actions, [True, True, True])
