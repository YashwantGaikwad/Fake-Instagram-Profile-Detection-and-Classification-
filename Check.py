from tkinter import *
def Train():
    """GUI"""
    import tkinter as tk
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

    root = tk.Tk()

    root.geometry("800x850+250+5")
    root.title("Instagram Fake Profile Detection")
    root.configure(background="#152238")
    
    profilepic = tk.IntVar()
    numsLengthusername = tk.DoubleVar()
    fullnamewords = tk.IntVar()
    numsLengthfullname = tk.DoubleVar()
    nameUsername = tk.IntVar()
    descriptionlength = tk.IntVar()
    private = tk.IntVar()
    posts = tk.IntVar()
    followers = tk.IntVar()
    follows = tk.IntVar()
    
    #===================================================================================================================
    def Detect():
        e1= profilepic.get()
        print(e1)
        e2= numsLengthusername.get()
        print(e2)
        e3= fullnamewords.get()
        print(e3)
        e4= numsLengthfullname.get()
        print(e4)
        e5= nameUsername.get()
        print(e5)
        e6= descriptionlength.get()
        print(e6)
        e7= private.get()
        print(e7)
        e8= posts.get()
        print(e8)
        e9= followers.get()
        print(e9)
        e10= follows.get()
        print(e10)
       
        
        #########################################################################################
        from joblib import dump ,load
        a1=load('svm.joblib')
        v= a1.predict([[e1,e2,e3,e4,e5,e6,e7,e8,e9,e10]])
        print(v)
        if v[0]==0:
            print("Fake")
            yes = tk.Label(root,text="Fake_Account",background="red",foreground="white",font=('times', 20, ' bold '),width=20)
            yes.place(x=250,y=650)
                     
        elif v[0]==1:
            print("Not Fake")
            no = tk.Label(root, text="Account_Not_Fake", background="green", foreground="white",font=('times', 20, ' bold '),width=20)
            no.place(x=250, y=650)
        
            


    l1=tk.Label(root,text="profilepic",background="white",font=('times', 20, ' bold '),width=20)
    l1.place(x=5,y=5)
    profilepic=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=profilepic)
    profilepic.place(x=400,y=5)

    l2=tk.Label(root,text="numsLengthusername",background="white",font=('times', 20, ' bold '),width=20)
    l2.place(x=5,y=50)
    numsLengthusername=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=numsLengthusername)
    numsLengthusername.place(x=400,y=50)

    l3=tk.Label(root,text="fullnamewords",background="white",font=('times', 20, ' bold '),width=20)
    l3.place(x=5,y=100)
    fullnamewords=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=fullnamewords)
    fullnamewords.place(x=400,y=100)
    
    l4=tk.Label(root,text="numsLengthfullname",background="white",font=('times', 20, ' bold '),width=20)
    l4.place(x=5,y=150)
    numsLengthfullname=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=numsLengthfullname)
    numsLengthfullname.place(x=400,y=150)

    l5=tk.Label(root,text="nameUsername",background="white",font=('times', 20, ' bold '),width=20)
    l5.place(x=5,y=200)
    nameUsername=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=nameUsername)
    nameUsername.place(x=400,y=200)
    
    l6=tk.Label(root,text="descriptionlength",background="white",font=('times', 20, ' bold '),width=20)
    l6.place(x=5,y=250)
    descriptionlength=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=descriptionlength)
    descriptionlength.place(x=400,y=250)
    
    l7=tk.Label(root,text="private",background="white",font=('times', 20, ' bold '),width=20)
    l7.place(x=5,y=300)
    private=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=private)
    private.place(x=400,y=300)
    
    l8=tk.Label(root,text="posts",background="white",font=('times', 20, ' bold '),width=20)
    l8.place(x=5,y=350)
    posts=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=posts)
    posts.place(x=400,y=350)
    
    
    l9=tk.Label(root,text="followers",background="white",font=('times', 20, ' bold '),width=20)
    l9.place(x=5,y=400)
    followers=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=followers)
    followers.place(x=400,y=400)
 
    l10=tk.Label(root,text="follows",background="white",font=('times', 20, ' bold '),width=20)
    l10.place(x=5,y=450)
    follows=tk.Entry(root,bd=2,width=20,font=("TkDefaultFont", 20),textvar=follows)
    follows.place(x=400,y=450)
 
    
    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=300,y=560)


    

    root.mainloop()

Train()