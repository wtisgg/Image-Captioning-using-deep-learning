import numpy as np
import string
from tkinter import *
from tkinter import filedialog
import PIL.Image
import PIL.ImageTk
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model
from pickle import load
from keras.applications.xception import Xception, preprocess_input
from keras.applications.vgg16 import VGG16
model = load_model("models/model_9.h5")
tokenizer = load(open("tokenizer.p", "rb"))
word_to_index = tokenizer.word_index
index_to_word = dict([index, word] for word, index in word_to_index.items())
vocab_size = len(tokenizer.word_index) + 1
max_len = 32

root = Tk()
root.title("Image Caption Generator")
root.state('zoomed')
root.resizable(width = True, height = True)
root.configure(bg="skyblue")

img1 = PIL.Image.open("logo.png")
img1 = img1.resize((80, 80))
img1 = PIL.ImageTk.PhotoImage(img1)
display_image1 = Label(root, image = img1)
display_image1.image = img1
display_image1.place(relx=0.25,rely=0.04)

panel1 = Label(root, text = 'Department of Computer Science and Engineering',fg="darkblue", bg="skyblue",font = ("Arial", 14))
panel1.place(relx = 0.33, rely = 0.05)

panel2 = Label(root, text = 'JNTUGV- University College of Engineering Vizianagaram',fg="darkblue",bg="skyblue", font = ("Arial", 14))
panel2.place(relx = 0.32, rely = 0.1)

panel3 = Label(root, text = 'Final Year Academic project',fg="red", bg="skyblue",font = ("Arial", 14))
panel3.place(relx = 0.4, rely = 0.15)

panel4 = Label(root, text = 'IMAGE CAPTION GENERATOR',bg="skyblue", font = ("Arial", 18))
panel4.place(relx = 0.37, rely = 0.2)

panel5 = Label(root, text = 'Done By-', fg="darkblue",bg="skyblue",font = ("Arial", 12))
panel5.place(relx = 0.89, rely = 0.78)

panel5 = Label(root, text = 'Ramya Sunkavalli', bg="skyblue",font = ("Arial", 12))
panel5.place(relx = 0.88, rely = 0.82)

panel6 = Label(root, text = 'D Usharani',bg="skyblue", font = ("Arial", 12))
panel6.place(relx = 0.88, rely = 0.86)

panel7 = Label(root, text = 'P Kavitha',bg="skyblue", font = ("Arial", 12))
panel7.place(relx = 0.88, rely = 0.9)

panel8 = Label(root, text = 'Sk Sameera',bg="skyblue", font = ("Arial", 12))
panel8.place(relx = 0.88, rely = 0.94)

filename = None
def chooseImage(event = None):
    global filename
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    img = img.resize((350, 300))
    img = PIL.ImageTk.PhotoImage(img)
    display_image = Label(root, image = img)
    display_image.image = img
    display_image.place(relx=0.37,rely=0.27)

value = StringVar()

def generateCaption(event = None):
    if(filename == None):
        value.set("No Image Selected")
    else:
        
        xcepmodel = Xception(include_top=False, pooling="avg")
        img = load_img(filename, target_size = (299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        img = img / 127.5
        img = img - 1.0
        features = xcepmodel.predict(img)

        #vggmodel=VGG16()
        #vggmodel = Model(inputs=model.inputs,outputs=model.layers[-2].output)
        #image = load_img(filename,target_size=(224,224))
        #image = img_to_array(image)
        #image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        
        #image = preprocess_input(image)
        #features = vggmodel.predict(image,verbose=0)
        in_text = 'start'
        for i in range(max_len):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=32)
            pred = model.predict([features,sequence], verbose=0)
            pred = np.argmax(pred)
            word = index_to_word[pred]
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        in_text = ' '.join(in_text.split(" ")[1: -1])
        text = in_text[0].upper() + in_text[1:] + '.'
        value.set(text)
    display_caption = Label(root, textvariable = value,fg='red', bg="skyblue",font=("Arial",18))
    display_caption.place(relx = 0.42, rely = 0.7)

button1 = Button(root, text='Choose an Image', font=(None, 18), activeforeground='red', bd=10, relief=RAISED, height=2, width=15, command = chooseImage) 
button1.place(relx = 0.3, rely = 0.8)
button2 = Button(root, text='Generate Caption', font=(None, 18), activeforeground = 'red', bd=10, relief=RAISED, height=2, width=15, command = generateCaption)
button2.place(relx = 0.56, rely = 0.8)
caption = Label(root, text='Caption : ', bg="skyblue",font=("Arial", 18))
caption.place(relx = 0.35, rely = 0.7)

root.mainloop()
