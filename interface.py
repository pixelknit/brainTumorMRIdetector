import tensorflow as tf
import cv2
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image

CATEGORIES = ["yes","no"]




class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Brain Tumor Detector")
        self.minsize(640, 400)

        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 1, row = 1, padx = 20, pady = 20)
        self.model = tf.keras.models.load_model("brain.model")
        self.filename = None

        self.button()


    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.runner)
        self.button.grid(column = 1, row = 1)


    def fileDialog(self):

        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetypes =
        (("jpeg files","*.jpg"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)

        self.img = Image.open(self.filename)
        photo = ImageTk.PhotoImage(self.img)

        self.label2 = Label(image=photo)
        self.label2.image = photo 
        self.label2.grid(column=1, row=4)
        
    def prepare(self,filepath):
        IMG_SIZE = 100
        img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

    def predict_text(self):
        prediction = self.model.predict([self.prepare(self.filename)])
        textto = CATEGORIES[int(prediction[0][0])]
        self.label3 = ttk.Label(text = "hola")
        self.label3.grid(column = 1, row = 5, padx=10, pady=10)
        if textto == "yes":
            self.label3.configure(text = "A tumor was detected in the patient's MRI scan",font=("Roboto",25))
        else:
            self.label3.configure(text = "A tumor was NOT detected in the patient's MRI scan", font=("Roboto",25))

    def runner(self):
        self.fileDialog()
        self.predict_text()

root = Root()
root.mainloop()


