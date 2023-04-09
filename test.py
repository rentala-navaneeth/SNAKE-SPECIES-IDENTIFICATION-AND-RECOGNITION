def testClassifier():
    text.delete('1.0', END)
    global filename, X, Y, X_train, y_train, X_test, y_test, classifier
    global labels
    if os.path.exists("model/model_weights.hdf5") == False:
        messagebox.showerror("Error", "No Model Found to Test")
    else:
        classifier = load_model("model/model_weights.hdf5")
    while True:
        filename = askopenfilename()
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img = cv2.imread(filename)
            img = cv2.resize(img, (80,80))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,80,80,3)
            y_pred = classifier.predict_classes(im2arr)
            text.insert(END,"Prediction : "+str(labels[y_pred[0]])+"\n")
            plt.imshow(img)
            plt.title("Prediction : "+str(labels[y_pred[0]]))
            plt.show()
            continue
        else:
            messagebox.showerror("Error", "Invalid Image Format")
            break
def getID(name):
    global labels
    return labels.index(name)

def showGraph():
    text.delete('1.0', END)
    global filename, X, Y, X_train, y_train, X_test, y_test, classifier
    global labels
    if os.path.exists("model/history.pckl") == False:
        messagebox.showerror("Error", "No History Found to Plot Graph")
    else:
        f = open('model/history.pckl', 'rb')
        hist = pickle.load(f)
        f.close()
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.title('Accuracy Graph')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def showConfusionMatrix():
    text.delete('1.0', END)
    global filename, X, Y, X_train, y_train, X_test, y_test, classifier
    global labels
    if os.path.exists("model/model_weights.hdf5") == False:
        messagebox.showerror("Error", "No Model Found to Create Confusion Matrix")
    else:
        classifier = load_model("model/model_weights.hdf5")
        predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, predict)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=labels, yticklabels=labels, title="Confusion matrix")
    plt.show()

    frame1 = tkinter.Frame(main)
    frame1.pack(side=TOP, pady=10)

    pathlabel = Label(frame1)
    pathlabel.pack()

    uploadButton = tkinter.Button(frame1, text="Upload", command=uploadDataset, padx=10)
    uploadButton.pack(side=LEFT, padx=10)

    processButton = tkinter.Button(frame1, text="Process", command=processDataset, padx=10)
    processButton.pack(side=LEFT, padx=10)

    trainButton = tkinter.Button(frame1, text="Train", command=trainDensenet, padx=10)
    trainButton.pack(side=LEFT, padx=10)

    testButton = tkinter.Button(frame1, text="Test", command=testClassifier, padx=10)
    testButton.pack(side=LEFT, padx=10)

    frame