from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


root = tk.Tk()
root.title("Fake Instagram Profile Prediction Using Machine Learning")


root.configure(background="purple")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

image = Image.open('s4.jpg')

image = image.resize((w, h))

background_image = ImageTk.PhotoImage(image)

background_image=ImageTk.PhotoImage(image)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=100, y=0) #, relwidth=1, relheight=1)

#img=ImageTk.PhotoImage(Image.open("s1.jpg"))

#img2=ImageTk.PhotoImage(Image.open("s2.jpg"))



logo_label=tk.Label()
logo_label.place(x=0,y=0)

x = 1




  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Fake Instagram Profile Prediction Using Machine Learning", font=('times', 35,' bold '), height=1, width=62,bg="purple",fg="white")
lbl.place(x=0, y=0)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Model_Training():
    data = pd.read_csv("insta_test.csv")
    data.head()
    data = data.dropna()

    """Feature Selection => Manual"""
    x = data.drop(['fake'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['fake']
    print(type(y))
    x.shape
    

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=12)


    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=200)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as svm.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=420)
    from joblib import dump
    dump (svcclassifier,"svm.joblib")
    print("Model saved as svm.joblib")

def Model_Training1():
    data = pd.read_csv("insta_test.csv")
    data = data.dropna()
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from joblib import dump


    x = data.drop(['fake'], axis=1)
    y = data['fake']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=111)

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=123)
    
    # Train the Random Forest classifier
    rf_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(x_test)
    print(y_pred)

    print("=" * 40)
    print("==========")
    print("Classification Report: ", classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy * 100)
    repo = (classification_report(y_test, y_pred))

    # Update the GUI code if needed
    label4 = tk.Label(root, text=str(repo), width=45, height=10, bg='khaki', fg='black', font=("Tempus Sanc ITC", 14))
    label4.place(x=305, y=200)
    
    label5 = tk.Label(root, text="Accuracy: " + str(ACC) + "%\nModel saved as random_forest.joblib", width=45, height=3, bg='khaki', fg='black', font=("Tempus Sanc ITC", 14))
    label5.place(x=305, y=420)

    # Save the trained Random Forest model
    dump(rf_classifier, "random_forest.joblib")
    print("Model saved as random_forest.joblib")
    
    
def Model_Training2():
    data = pd.read_csv("insta_test.csv")
    data = data.dropna()
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from joblib import dump


    x = data.drop(['fake'], axis=1)
    y = data['fake']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=12)

    # Create a Decision Tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    
    # Train the Decision Tree classifier
    dt_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = dt_classifier.predict(x_test)
    print(y_pred)

    print("=" * 40)
    print("==========")
    print("Classification Report: ", classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy * 100)
    repo = (classification_report(y_test, y_pred))

    # Update the GUI code if needed
    label4 = tk.Label(root, text=str(repo), width=45, height=10, bg='khaki', fg='black', font=("Tempus Sanc ITC", 14))
    label4.place(x=305, y=200)
    
    label5 = tk.Label(root, text="Accuracy: " + str(ACC) + "%\nModel saved as decision_tree.joblib", width=45, height=3, bg='khaki', fg='black', font=("Tempus Sanc ITC", 14))
    label5.place(x=305, y=420)

    # Save the trained Decision Tree model
    dump(dt_classifier, "decision_tree.joblib")
    print("Model saved as decision_tree.joblib")

#def call_file():
   # import Check_carrier
   # Check_carrier.Train()

def call_file():
   from subprocess import call
   call(['python','Check.py'])



def window():
    root.destroy()

# button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
#                     text="Data_Preprocessing", command=Data_Preprocessing, width=15, height=2)
# button2.place(x=5, y=120)

button3 = tk.Button(root, foreground="white", background="#152238", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model_SVM", command=Model_Training, width=15, height=2)
button3.place(x=5, y=200)

button5 = tk.Button(root, foreground="white", background="#152238", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model_RF", command=Model_Training1, width=15, height=2)
button5.place(x=5, y=300)

button6 = tk.Button(root, foreground="white", background="#152238", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model_DT", command=Model_Training2, width=15, height=2)
button6.place(x=5, y=400)

button4 = tk.Button(root, foreground="white", background="#152238", font=("Tempus Sans ITC", 14, "bold"),
                    text="Check", command=call_file, width=15, height=2)
button4.place(x=5, y=500)

exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=600)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''