# Program to make a simple login screen

import os
import cv2
import numpy as np
from skimage import morphology, measure
from skimage.color import label2rgb
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from numpy import result_type
from signature import extract, validate, crop_validate, mult_validate
from gabor_v2 import GT_similarity

# Mach Threshold
THRESHOLD = 50

root=tk.Tk()
root.title("Signature Verification")

# setting the windows size
root.geometry("600x400")

# declaring string variable
# for storing name and password
name_var=tk.StringVar()
passw_var=tk.StringVar()


# defining a function that will
# get the name and password and
# print them on the screen
def submit():
    name = name_var.get()
    # password=passw_var.get()
    
    print("\n\n The name is : " + name)
    # print("The password is : " + password)

    for i in range(4):
        file = 'assets/' + name + '/' + name + str(i+1) + '.jpg'
        s_file = 'temp/' + name + str(i+1) + '.jpg'
        print("\n", file, "\n")
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        image = cv2.resize(morph, (300, 300))

        cv2.imwrite(s_file, image)

        cv2.imshow(str(i+1), image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    name_var.set("")
    passw_var.set("")


def browsefunc(ent):
    filename = askopenfilename(filetypes=([
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg"),
    ]))
    ent.delete(0, tk.END)
    ent.insert(tk.END, filename)  # add this


def capture_image_from_cam_into_temp(sign=1):
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cv2.namedWindow("test")

    # img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            if not os.path.isdir('temp'):
                os.mkdir('temp', mode=0o777)  # make sure the directory exists
            # img_name = "./temp/opencv_frame_{}.png".format(img_counter)
            if(sign == 1):
                img_name = "./temp/test_img1.png"
            else:
                img_name = "./temp/test_img2.png"
            print('imwrite=', cv2.imwrite(filename=img_name, img=frame))
            print("{} written!".format(img_name))
            # img_counter += 1
    cam.release()
    cv2.destroyAllWindows()
    return True


def captureImage(ent, sign=1):
    if(sign == 1):
        filename = os.getcwd()+'\\temp\\test_img1.png'
    else:
        filename = os.getcwd()+'\\temp\\test_img2.png'
    # messagebox.showinfo(
    #     'SUCCESS!!!', 'Press Space Bar to click picture and ESC to exit')
    res = None
    res = messagebox.askquestion(
        'Click Picture', 'Press Space Bar to click picture and ESC to exit')
    if res == 'yes':
        capture_image_from_cam_into_temp(sign=sign)
        ent.delete(0, tk.END)
        ent.insert(tk.END, filename)
    return True


def checkSimilarity(window, path1, path2):
    # result = mult_validate(path1=path1, path2=path2)
    result = GT_similarity(path1=path1, path2=path2)
    if(result < THRESHOLD):
        messagebox.showerror("Failure: Signatures Do Not Match",
                             "Signatures are "+str(result)+f" % similar!!")
        pass
    else:
        messagebox.showinfo("Success: Signatures Match",
                            "Signatures are "+str(result)+f" % similar!!")

    cv2.destroyAllWindows()

    return True


def extractor(window, path1, path2):
    extract(path1=path1, path2=path2)
    return True


uname_label = tk.Label(root, text="Validate User Signature:", font=('calibre',14,'bold'))
uname_label.place(x=180, y=50)

# creating a label for name using widget Label
name_label = tk.Label(root, text = 'Username', font=('calibre',10,'bold'))
name_label.place(x=60, y=120)

# creating a entry for input name using widget Entry
name_entry = tk.Entry(root,textvariable = name_var, font=('calibre',10,'normal'))
name_entry.place(x=200, y=120)

# creating a button using the widget button that will call the submit function
sub_btn = tk.Button(root, text = 'View Data', font=('calibre',10,'normal'), command = submit)
sub_btn.place(x=380, y=120)


# Image Message
img_message = tk.Label(root, text="Submit Signature", font=('calibre',10,'bold'))
img_message.place(x=60, y=180)

# Image Submit
image_path_entry = tk.Entry(root, font=('calibre',10,'normal'))
image_path_entry.place(x=200, y=180)

# Capture Button
img_browse_button = tk.Button(
    root, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=image_path_entry))
img_browse_button.place(x=380, y=180)

# Or Message
or_message = tk.Label(root, text="or", font=('calibre',10,'bold'))
or_message.place(x=450, y=183)

# Browse Button
img_capture_button = tk.Button(
    root, text="Capture", font=('calibre',10,'normal'), command=lambda: captureImage(ent=image_path_entry, sign=2))
img_capture_button.place(x=480, y=180)



# extract_button = tk.Button(
#     root, text="Extract", font=('calibre',12,'bold'), command=lambda: extractor(window=root,
#                                                                    path1=name_entry.get(),
#                                                                    path2=image_path_entry.get(),))
# extract_button.place(x=240, y=240)

validate_button = tk.Button(
    root, text="Validate", font=('calibre',12,'bold'), command=lambda: checkSimilarity(window=root,
                                                                   path1=name_entry.get(),
                                                                   path2=image_path_entry.get(),))
validate_button.place(x=240, y=240)


# # creating a label for password
# passw_label = tk.Label(root, text = 'Password', font = ('calibre',10,'bold'))
  
# # creating a entry for password
# passw_entry=tk.Entry(root, textvariable = passw_var, font = ('calibre',10,'normal'), show = '*')

# performing an infinite loop for the window to display
root.mainloop()
