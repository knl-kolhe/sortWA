# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:42:30 2020

@author: Kunal
"""

from keras.models import load_model
import os
import cv2
import numpy as np
import keras
import tensorflow as tf



import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog


class sortWA:
    def __init__(self):
        self.ui = tk.Tk()
        self.ui.title("Sort Whatsapp Images")
        
        self.ui.sourceFolder = ''
        self.ui.model_path = ''
        self.ui.label_path = ''
        
        folderVar = tk.StringVar()
        folderText = tk.Label(self.ui, textvariable = folderVar)
        folderVar.set("Get Path to Images Folder")
        folderText.grid(row=0, column=0, padx=(50,10), pady=20)
        
        self.folderEntry = tk.Entry(self.ui, width=50)
        self.folderEntry.grid(row=0, column = 1, padx=(10,10), pady=20)
            
        getFolder = tk.Button(self.ui, text='Get folder path', command=self.getFolderPath, width=20)
        getFolder.grid(row=0, column=2, padx=(10,50), pady=20)
        
        # ---------------------------------------------------------------------------
        modelVar = tk.StringVar()
        modelText = tk.Label(self.ui, textvariable = modelVar)
        modelVar.set("Get filename of sorting model (.h5)")
        modelText.grid(row=1, column=0, padx=(50,10), pady=20)
        
        self.modelEntry = tk.Entry(self.ui, width=50)
        self.modelEntry.grid(row=1, column = 1, padx=(10,10), pady=20)
            
        getModel = tk.Button(self.ui, text='Get model path', command=self.getModelPath, width=20)
        getModel.grid(row=1, column=2, padx=(10,50), pady=20)
        
        # ----------------------------------------------------------------------------
        labelVar = tk.StringVar()
        labelText = tk.Label(self.ui, textvariable = labelVar)
        labelVar.set("Get Path to Images Folder")
        labelText.grid(row=2, column=0, padx=(50,10), pady=20)
        
        self.labelEntry = tk.Entry(self.ui, width=50)
        self.labelEntry.grid(row=2, column = 1, padx=(10,10), pady=20)
            
        getLabel = tk.Button(self.ui, text='Get Label path', command=self.getLabelPath, width=20)
        getLabel.grid(row=2, column=2, padx=(10,50), pady=20)
        
        # ----------------------------------------------------------------------------
        sortImagesButton = tk.Button(self.ui, text='Sort all images into folders', command=self.sortFolder, width=20)
        sortImagesButton.grid(row=3, column=0, padx=(10,50), pady=20)
        
        #--------------------------------------------------------------------------
        quitButton = tk.Button(self.ui, text='Quit', command=self.quit, width=20)
        quitButton.grid(row=3, column=2, padx=(10,50), pady=20)
        
    def quit(self):
        self.ui.quit()
        self.ui.destroy()
        
    
    def parse_dirs(self):
        with open(self.label_path) as label_file:
            labels = label_file.read()
            labels = labels.split("\n")
            # print(labels)
        return labels  
    
    def create_sortDirs(self):
        for i in range(len(self.dirs)):
            if not os.path.exists(os.path.join(self.folder_path, self.dirs[i])):
                os.makedirs(os.path.join(self.folder_path, self.dirs[i]))
    
    def load_imagesPaths(self):
        self.images_list = []
        
        for (dirpath, dirnames, filenames) in os.walk(self.folder_path):
            self.images_list.extend(filenames)
            break
        
        if len(self.images_list)==0:
            tk.messagebox.showerror(title="Empty Folder", message="Images Folder does not contain any images")
            return False
        
        return True
    
    def process_images(self, number=5000):       
        total_imgs = len(self.images_list)
        if total_imgs<number:
            number = total_imgs
        
        for i in range(int(np.ceil(total_imgs/number))):
            start = i*number
            end = (i+1)*number
            if end>total_imgs:
                end = total_imgs
                
            all_imgs = []
            
            for filename in self.images_list[start:end]:
                img = cv2.imread(os.path.join(self.folder_path, filename))
                img = cv2.resize(img,(224,224))
                img = img / 255.0
                all_imgs.append(img)
                
            self.processed_images = np.array(all_imgs)
            
            self.sortBatch(self.images_list[start:end])
        tk.messagebox.showinfo(title="Sorting Successful", message="All the images have been sorted")
            
    def sortBatch(self, batch_images_list):
        # start = timer()
        categories = np.argmax(self.sortWA_model.predict(self.processed_images),axis=1)
        # end = timer()
        # print(f"Time elapsed for sorting all images is {end - start}, so we can sort {all_imgs.shape[0]/(end-start)} images per second")
        
        for filename, category in zip(batch_images_list,categories):
            os.rename(os.path.join(self.folder_path,filename),os.path.join(self.folder_path, self.dirs[category], filename))

    def sortAllImages(self, folder_path, model_path, label_path):
        self.folder_path = folder_path
        self.model_path = model_path
        self.label_path = label_path
        
        self.sortWA_model = load_model(self.model_path)
        
        self.dirs = self.parse_dirs()
        # print(self.dirs)
        # print(len(self.sortWA_model.outputs))
        model_output_shape = self.sortWA_model.outputs[0].shape[1]
        # print(self.sortWA_model.output_shapes)
        
        if len(self.dirs) != model_output_shape:
            tk.messagebox.showerror(title="Wrong label file", message="Number of categories in label file does not match outputs for model")
            return
        
        
        self.create_sortDirs()
            
        
        if(self.load_imagesPaths()):
            self.process_images(5)
    
    def getFolderPath(self):  
        self.ui.sourceFolder =  filedialog.askdirectory(parent=self.ui, initialdir= "./", title='Please select the directory where all Images are stored')
        # self.ui.sourceFolder =  "C:/!Kunal/Projects/SortWhatsapp/test"
        self.folderEntry.insert(0, self.ui.sourceFolder)
        
    
    def getModelPath(self):  
        self.ui.model_path = filedialog.askopenfilename(parent=self.ui, initialdir= "./", title='Please select ML model for sorting (.h5)')
        # self.ui.model_path =  "C:/!Kunal/Projects/SortWhatsapp/model-best.h5"
        self.modelEntry.insert(0, self.ui.model_path)        
        print(self.ui.model_path)
        
    def getLabelPath(self):  
        self.ui.label_path =  filedialog.askopenfilename(parent=self.ui, initialdir= "./", title='Please select categories labels')
        # self.ui.label_path =  "C:/!Kunal/Projects/SortWhatsapp/sortWA_categories.txt"
        self.labelEntry.insert(0, self.ui.label_path)
        
        
        
    def sortFolder(self):
        if self.ui.sourceFolder=='' or self.ui.model_path=='' or self.ui.label_path=='':
            tk.messagebox.showerror(title="Set Path", message="Enter all required paths")
        elif not (os.path.isdir(self.ui.sourceFolder)):
            tk.messagebox.showerror(title="Folder Path does not exist", message="Enter correct folder path")
        elif not (os.path.isfile(self.ui.model_path)):
            tk.messagebox.showerror(title="Model path does not exist", message="Enter correct model path")
        elif not (os.path.isfile(self.ui.label_path)):
            tk.messagebox.showerror(title="Label path does not exist", message="Enter correct label path")
        else:
            self.sortAllImages(self.ui.sourceFolder, self.ui.model_path, self.ui.label_path)
    
    def run(self):        
            self.ui.mainloop()
        
            
    
    
def main():    
    sortWA_obj = sortWA()
    try:
        sortWA_obj.run()
    except:
           tk.messagebox.showerror(title="Error", message="Something went wrong")   
        
        
if __name__ == "__main__":
    main()
        
















# main_win = tkinter.Tk()
# main_win.geometry("500x300")
# main_win.sourceFolder = ''
# e1 = tk.Entry(master)

# main_win.sourceFile = ''
# def chooseDir():
#     main_win.sourceFolder =  filedialog.askdirectory(parent=main_win, initialdir= "./", title='Please select a directory')

# b_chooseDir = tkinter.Button(main_win, text = "Chose Folder", width = 20, height = 3, command = chooseDir)
# b_chooseDir.place(x = 50,y = 50)
# b_chooseDir.width = 100


# def chooseFile():
#     main_win.sourceFile = filedialog.askopenfilename(parent=main_win, initialdir= "/", title='Please select a directory')

# b_chooseFile = tkinter.Button(main_win, text = "Chose File", width = 20, height = 3, command = chooseFile)
# b_chooseFile.place(x = 250,y = 50)
# b_chooseFile.width = 100

# main_win.mainloop()
# print(main_win.sourceFolder)
# print(main_win.sourceFile )


'''
folder_path = ui.sourceFolder#"C:\!Kunal\Projects\SortWhatsapp\WithoutFaces"

sortWA_model = load_model("./model-best.h5")

dirs = ["Delete",  "Documents", "Objects", "People", "Scenery", "TextFace"]
for i in range(len(dirs)):
    if not os.path.exists(os.path.join(folder_path, dirs[i])):
        os.makedirs(os.path.join(folder_path, dirs[i]))


images_list = []

for (dirpath, dirnames, filenames) in os.walk(folder_path):
    images_list.extend(filenames)
    break

all_imgs = []
start = timer()
for filename in images_list:
    # print(os.path.join(folder_path, filename))
    
    img = cv2.imread(os.path.join(folder_path, filename))
    # imgplot = plt.imshow(img)
    # plt.show()
    img = cv2.resize(img,(224,224))
    img = img / 255.0
    all_imgs.append(img)
    # img = np.expand_dims(img, axis=0)
    # print(img.shape)
    # img = tf.convert_to_tensor(
    #     img, dtype=None, dtype_hint=None, name=None
    # )
    # category = np.argmax(sortWA_model.predict(img))
    # os.rename(os.path.join(folder_path,filename),os.path.join(folder_path,dirs[category],filename))
    
    # print(f"Time elapsed for sorting 1 image is {end - start}, so we can sort {1/(end-start)} images per second")
    # break
all_imgs = np.array(all_imgs)
end = timer()
print(f"time to convert all image to np array is {end-start}, per second we process {all_imgs.shape[0]/(end-start)}")


start = timer()
categories = np.argmax(sortWA_model.predict(all_imgs),axis=1)
end = timer()
print(f"Time elapsed for sorting all images is {end - start}, so we can sort {all_imgs.shape[0]/(end-start)} images per second")

for filename, category in zip(images_list,categories):
    os.rename(os.path.join(folder_path,filename),os.path.join(folder_path,dirs[category],filename))
'''
