# Live Sign Language Translator
Sign language is a form of manual communication that is commonly used by deaf people. This project is a beginner's attempt in demonstrating the use of deep learning for real world problems. 
<br/>
![demo](https://user-images.githubusercontent.com/67902015/152925333-8fdfeab3-2218-4196-b091-82208350bdf6.gif)
<br/>

This implementation is technically a live **hand-shoulder gesture recognizer** because it can be trained to recognize any gesture. But, its main purpose is for translating sign language.


## How it works
Using the [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) model, **134** keypoints are extracted from the person's **hands** and **shoulders**. This data is then used to train a basic model that can identify multiple gestures. 

## How to Use

### Running 
* When creating a new **ASLTranslator** instance, it is required to define the ***model*** and ***list of words*** that will be used. Sample _model.h5_ and _word_list.txt_ can be found in this repo.
* Use the ASLTranslator's **StartCapture** method to start video capture.

```python
aslt = ASLTranslator(MODEL_PATH, word_list)
aslt.StartCapture()
```
### Training
**ASLTranslator** can be trained to recognize multiple gestures and translate them on the go. Here are the steps for training new data:
1. Create a folder where you want to save your data for training.
2. Start collecting data using the **CollectDataforWord** method. This method will automatically save the keypoints for each frame of a sample in the specified directory.
3. You can use the **GetNpyDataFromPath** method to load those keypoints.
```python
for word in word_list:
  aslt.CollectDataforWord(word, DATA_PATH, sampleCount = 10) #save keypoints data for each word in the list
sequences, labels = aslt.GetNpyDataFromPath(DATA_PATH) #load keypoints data
```
4. Create and train your new model using the following methods: **CreateModel, SplitData & TrainModel**.
```python
model = aslt.CreateModel()
model_name = 'new_model.h5'
x_train, x_test, y_train, y_test = aslt.SplitData(sequences, labels)
aslt.TrainModel(model, x_train, y_train, epochs = 100, model_name = model_name)
```
You can check [tester.py](https://github.com/lloyd-axe/Live-sign-language-translator/blob/main/tester.py) for reference.
## Installation
Installation is pretty straight forward. Clone the repo.
```git
git clone https://github.com/lloyd-axe/Live-sign-language-translator
```
### Website Installation
1. Open [Website](https://github.com/lloyd-axe/Live-sign-language-translator/tree/main/Website) folder.
2. Install packages listed in its [requirements.txt](https://github.com/lloyd-axe/Live-sign-language-translator/blob/main/Website/requirements.txt).
```
pip install -r requirements.txt
```
3. Change **SECRET_KEY** in [settings.py](https://github.com/lloyd-axe/Live-sign-language-translator/blob/main/Website/project_sigua/project_sigua/settings.py).
4. Attach **model.h5** and **word_list.txt** files to **Website/project_sigua/static/** folder
5. Run server.

```
python manage.py runserver
```

## Possible Improvements
* Use other techniques like convolutional neural network
* Larger training datasets
