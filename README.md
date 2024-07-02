# website_chatbot
This project demonstrates the creation of a simple chatbot using TensorFlow, Keras, and other essential libraries. The chatbot is trained using predefined intents and responses, making it capable of understanding and responding to basic user inputs.

# Table of Contents

	•	Installation
	•	Usage
	•	Dataset
	•	Model Training
	•	Testing the Chatbot
	•	Contributing
	•	License 

# Installation
1.Clone the repository:
    
    git clone https://github.com/your-username/chatbot.git

2.Install the required packages:

    pip install -r requirements.txt

The main packages include:
	•	TensorFlow
	•	NumPy
	•	scikit-learn
	•	colorama

 Usage

1.	Ensure you have the necessary files:
•	intents.json (defines the intents and responses)
•	chatbot.ipynb (the main code for training and testing the chatbot)

2.	To train the model, run the notebook or script:

        jupyter notebook chatbot.ipynb

# Dataset
The dataset used for training the chatbot is defined in intents.json. Here is an example of the structure:

    {
       "intents": [
              {
                  "tag": "greeting",
                  "patterns": ["Hi", "Hey", "Is anyone there?", "Hello", "Hay"],
                  "responses": ["Hello", "Hi", "Hi there"]
              },
             ...
       ]
    }

# Model Training 
The model is built and trained using TensorFlow and Keras. Here is a summary of the process:

1.	Load and preprocess data: The intents are loaded from intents.json, and the sentences and labels are extracted and encoded.
2.	Tokenization and padding: The sentences are tokenized and padded to ensure uniform input length.
3.	Model creation: A sequential model is created with an embedding layer, a global average pooling layer, and dense layers with ReLU activation.
4.	Model training: The model is compiled using sparse categorical cross-entropy loss and the Adam optimizer, and trained for 500 epochs.
5.	Model saving: The trained model, tokenizer, and label encoder are saved for later use.

# Testing the Chatbot
To test the chatbot, load the trained model and other necessary objects, and run the chat() function:

    import json
    import numpy as np
    from tensorflow import keras
    from sklearn.preprocessing import LabelEncoder
    import colorama
    colorama.init()
    from colorama import Fore, Style, Back
    import random
    import pickle

    with open("intents.json") as file:
        data = json.load(file)

    def chat():
        model = keras.models.load_model('chat_model')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)
        max_len = 20
        while True:
            print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
            inp = input()
            if inp.lower() == "quit":
                break
            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                            truncating='post', maxlen=max_len))
            tag = lbl_encoder.inverse_transform([np.argmax(result)])
            for i in data['intents']:
                if i['tag'] == tag:
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

    print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
    chat()




    

