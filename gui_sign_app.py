import os
import cv2
import time
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image

from gui_gabor import GT_similarity



path = os.getcwd()
d = 'registered'
files = os.path.join(path, d)
isdir = os.path.isdir(files)
if not isdir:
    os.mkdir(files)

d = 'temp'
files = os.path.join(path, d)
isdir = os.path.isdir(files)
if not isdir:
    os.mkdir(files)



class Application(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)

        self.pack()
        self.create_front()


    def create_front(self):

        def gui_destroy():
            root.destroy()

        self.uname_label = tk.Label(root, text="Signature Verification System", font=('calibre',14,'bold'), bg='medium aquamarine')
        self.uname_label.place(x=110, y=40)

        self.go_upload = tk.Button(
            root, text="Register a Signature", font=('calibre',12,'bold'), bg='skyblue2', command=self.create_upload)
        self.go_upload.place(x=160, y=100)


        self.go_verify = tk.Button(
            root, text="Verify a Signature", font=('calibre',12,'bold'), bg='gold2', command=self.create_verify)
        self.go_verify.place(x=170, y=160)


        quit = tk.Button(root, text="Exit", font=('calibre',12,'bold'), bg = "tomato", width=5, command=root.destroy)
        quit.place(x=220, y=220)


    def create_upload(self):

        def image_save(window, path0, path1, path2, path3, path4, path5):
            xx = self.name_entry.get()
            print("\n User name is : " + xx + "\n")

            sign_list = [path1, path2, path3, path4, path5]

            if path0=='':
                messagebox.showerror("Warning!", "Username can not be empty!")

            else:
                ff = 0
                for ee in sign_list:
                    file_exists = os.path.exists(ee)

                    if not file_exists:
                        messagebox.showerror("Warning!",
                                            "You must upload all 5 signatures!")
                        break
                    else:
                        ff += 1

                if ff==5:
                    cc = 0
                    for i in sign_list:
                        file = i
                        cc += 1
                        s_file = 'registered/' + xx + str(cc) + '.jpg'
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

                    #     cv2.imshow(str(cc), image)
                    #     cv2.waitKey(0)

                    # cv2.destroyAllWindows()

                    dataset = tk.Label(root_u, text="Display Registered Signatures", font=('calibre',12,'bold'))
                    dataset.place(x=130, y=460)
                    
                    s_file1 = './registered/' + path0 + '1.jpg'
                    img1 = Image.open(s_file1)
                    img1 = img1.resize((80, 80), Image.ANTIALIAS)
                    img1 = ImageTk.PhotoImage(img1)
                    panel1 = tk.Label(root_u, image=img1)
                    panel1.place(x=20, y=500)

                    s_file2 = './registered/' + path0 + '2.jpg'
                    img2 = Image.open(s_file2)
                    img2 = img2.resize((80, 80), Image.ANTIALIAS)
                    img2 = ImageTk.PhotoImage(img2)
                    panel2 = tk.Label(root_u, image=img2)
                    panel2.place(x=110, y=500)

                    s_file3 = './registered/' + path0 + '3.jpg'
                    img3 = Image.open(s_file3)
                    img3 = img3.resize((80, 80), Image.ANTIALIAS)
                    img3 = ImageTk.PhotoImage(img3)
                    panel3 = tk.Label(root_u, image=img3)
                    panel3.place(x=200, y=500)

                    s_file4 = './registered/' + path0 + '4.jpg'
                    img4 = Image.open(s_file4)
                    img4 = img4.resize((80, 80), Image.ANTIALIAS)
                    img4 = ImageTk.PhotoImage(img4)
                    panel4 = tk.Label(root_u, image=img4)
                    panel4.place(x=290, y=500)

                    s_file5 = './registered/' + path0 + '5.jpg'
                    img5 = Image.open(s_file5)
                    img5 = img5.resize((80, 80), Image.ANTIALIAS)
                    img5 = ImageTk.PhotoImage(img5)
                    panel5 = tk.Label(root_u, image=img5)
                    panel5.place(x=380, y=500)


                    messagebox.showinfo("Success!",
                                            "Signatures are registered!!")
                    
                    dataset.destroy()

        def check_data():
            xx = self.name_entry.get()
            print("\n User name is : " + xx + "\n")

            file = 'registered/' + xx + '1.jpg'
            # print("\n", file, "\n")
            file_exists = os.path.exists(file)

            if not file_exists:
                messagebox.showwarning("Checked!",
                                    "User doesn't exist! Please Continue Uploading for Registration!")
            else:
                messagebox.showinfo("Checked!",
                                    "User exists! You can exit to Verify or Continue Upload and Update!")

        def browsefunc(ent):
            filename = askopenfilename(filetypes=([
                ("image", ".jpeg"),
                ("image", ".png"),
                ("image", ".jpg"),
            ]))
            ent.delete(0, tk.END)
            ent.insert(tk.END, filename)  # add this

        def gui_destroy():
            root_u.destroy()


        root_u = tk.Toplevel(self)
        root_u.title('Signature Registration')
        root_u.geometry('500x600+650+50')

        self.uname_label = tk.Label(root_u, text="Register User Signature", font=('calibre',14,'bold'), bg='skyblue2')
        self.uname_label.place(x=135, y=25)

        # creating a label for name using widget Label
        self.name_label = tk.Label(root_u, text = 'Username:', font=('calibre',10,'bold'))
        self.name_label.place(x=60, y=90)

        # creating a entry for input name using widget Entry
        self.name_entry = tk.Entry(root_u, bd=3, font=('calibre',10,'normal'))
        self.name_entry.place(x=170, y=90)

        # creating a button using the widget button that will call the submit function
        sub_btn = tk.Button(root_u, text = 'Check Data', font=('calibre',10,'normal'), command = check_data)
        sub_btn.place(x=350, y=88)


        # Image 1
        self.img_message = tk.Label(root_u, text="Signature 1:", font=('calibre',10,'bold'))
        self.img_message.place(x=60, y=140)
        # Image Submit
        self.image_path_entry1 = tk.Entry(root_u, bd=3, font=('calibre',10,'normal'))
        self.image_path_entry1.place(x=170, y=140)
        # Browse Button
        self.img_browse_button = tk.Button(
            root_u, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=self.image_path_entry1))
        self.img_browse_button.place(x=350, y=138)


        # Image 2
        self.img_message = tk.Label(root_u, text="Signature 2:", font=('calibre',10,'bold'))
        self.img_message.place(x=60, y=180)
        # Image Submit
        self.image_path_entry2 = tk.Entry(root_u, bd=3, font=('calibre',10,'normal'))
        self.image_path_entry2.place(x=170, y=180)
        # Browse Button
        self.img_browse_button = tk.Button(
            root_u, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=self.image_path_entry2))
        self.img_browse_button.place(x=350, y=178)


        # Image 3
        self.img_message = tk.Label(root_u, text="Signature 3:", font=('calibre',10,'bold'))
        self.img_message.place(x=60, y=220)
        # Image Submit
        self.image_path_entry3 = tk.Entry(root_u, bd=3, font=('calibre',10,'normal'))
        self.image_path_entry3.place(x=170, y=220)
        # Browse Button
        self.img_browse_button = tk.Button(
            root_u, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=self.image_path_entry3))
        self.img_browse_button.place(x=350, y=218)


        # Image 4
        self.img_message = tk.Label(root_u, text="Signature 4:", font=('calibre',10,'bold'))
        self.img_message.place(x=60, y=260)
        # Image Submit
        self.image_path_entry4 = tk.Entry(root_u, bd=3, font=('calibre',10,'normal'))
        self.image_path_entry4.place(x=170, y=260)
        # Browse Button
        self.img_browse_button = tk.Button(
            root_u, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=self.image_path_entry4))
        self.img_browse_button.place(x=350, y=258)


        # Image 5
        self.img_message = tk.Label(root_u, text="Signature 5:", font=('calibre',10,'bold'))
        self.img_message.place(x=60, y=300)
        # Image Submit
        self.image_path_entry5 = tk.Entry(root_u, bd=3, font=('calibre',10,'normal'))
        self.image_path_entry5.place(x=170, y=300)
        # Browse Button
        self.img_browse_button = tk.Button(
            root_u, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=self.image_path_entry5))
        self.img_browse_button.place(x=350, y=298)



        # registered Button
        self.register_button = tk.Button(
            root_u, text="Register", font=('calibre',12,'bold'), bg='gold2', command=lambda: image_save(window=root_u,
                                                                        path0=self.name_entry.get(),
                                                                        path1=self.image_path_entry1.get(),
                                                                        path2=self.image_path_entry2.get(),
                                                                        path3=self.image_path_entry3.get(),
                                                                        path4=self.image_path_entry4.get(),
                                                                        path5=self.image_path_entry5.get(),), width=8)
        self.register_button.place(x=198, y=350)


        # Exit Button
        go_exit = tk.Button(
            root_u, text="Exit", font=('calibre',12,'bold'), bg='tomato', command=lambda: gui_destroy(), width=5)
        go_exit.place(x=214, y=400)

        root_u.mainloop()


    def create_verify(self):
        # Mach Threshold
        THRESHOLD = 50

        root_v=tk.Toplevel(self)
        root_v.title("Signature Verification")

        # setting the windows size
        root_v.geometry("500x600+650+50")


        # defining a function that will get the name and password and print them on the screen
        def view_data():
            name = self.name_entry.get()
            
            print("\n The name is : " + name + "\n")

            for i in range(5):
                file = 'registered/' + name + str(i+1) + '.jpg'
                print("\n", file, "\n")
                image = cv2.imread(file)

                image = cv2.resize(image, (300, 300))

                cv2.imshow(str(i+1), image)
                cv2.waitKey(0)

            cv2.destroyAllWindows()


        def check_data():
            name = self.name_entry.get()
            print("\n User name is : " + name + "\n")

            file = 'registered/' + name + '1.jpg'
            # print("\n", file, "\n")
            file_exists = os.path.exists(file)

            if not file_exists:
                messagebox.showerror("Warning!",
                                    "User doesn't exist! Please Enter Correct Username!")
            else:
                messagebox.showinfo("Checked!",
                                    "User exists! Please Continue Upload to Verify!")


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


        def checkSimilarity(window, path0, path1):

            if path0=='' or path1=='':
                messagebox.showerror("Warning!", "Username or Uploaded Image can not be empty while varifying!")

            else:
                ch_file = './registered/' + path0 + '1.jpg'
                file_exists = os.path.exists(ch_file)

                if not file_exists:
                    messagebox.showerror("Warning!", "User does not exist in Database! Please enter Username correctly for verifying! Or, Exit and Go to User Registration")

                else:
                    dataset = tk.Label(root_v, text="Display Images in Database", font=('calibre',12,'bold'))
                    dataset.place(x=140, y=300)
                    
                    s_file1 = './registered/' + path0 + '1.jpg'
                    img1 = Image.open(s_file1)
                    img1 = img1.resize((80, 80), Image.ANTIALIAS)
                    img1 = ImageTk.PhotoImage(img1)
                    panel1 = tk.Label(root_v, image=img1)
                    panel1.place(x=20, y=340)

                    s_file2 = './registered/' + path0 + '2.jpg'
                    img2 = Image.open(s_file2)
                    img2 = img2.resize((80, 80), Image.ANTIALIAS)
                    img2 = ImageTk.PhotoImage(img2)
                    panel2 = tk.Label(root_v, image=img2)
                    panel2.place(x=110, y=340)

                    s_file3 = './registered/' + path0 + '3.jpg'
                    img3 = Image.open(s_file3)
                    img3 = img3.resize((80, 80), Image.ANTIALIAS)
                    img3 = ImageTk.PhotoImage(img3)
                    panel3 = tk.Label(root_v, image=img3)
                    panel3.place(x=200, y=340)

                    s_file4 = './registered/' + path0 + '4.jpg'
                    img4 = Image.open(s_file4)
                    img4 = img4.resize((80, 80), Image.ANTIALIAS)
                    img4 = ImageTk.PhotoImage(img4)
                    panel4 = tk.Label(root_v, image=img4)
                    panel4.place(x=290, y=340)

                    s_file5 = './registered/' + path0 + '5.jpg'
                    img5 = Image.open(s_file5)
                    img5 = img5.resize((80, 80), Image.ANTIALIAS)
                    img5 = ImageTk.PhotoImage(img5)
                    panel5 = tk.Label(root_v, image=img5)
                    panel5.place(x=380, y=340)


                    d_result = tk.Label(root_v, text="Verification Result", font=('calibre',13,'bold'))
                    d_result.place(x=20, y=450)
                    
                    in_image = cv2.imread(path1)
                    s_file6 = 'temp/verifying.jpg'

                    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
                    divide = cv2.divide(gray, blur, scale=255)
                    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    in_image = cv2.resize(morph, (300, 300))
                    cv2.imwrite(s_file6, in_image)

                    img0 = Image.open(s_file6)
                    img0 = img0.resize((80, 80), Image.ANTIALIAS)
                    img0 = ImageTk.PhotoImage(img0)
                    panel0 = tk.Label(root_v, image=img0)
                    panel0.place(x=110, y=490)

                    # result = mult_validate(path1=path1, path2=path2)
                    result = GT_similarity(path0=path0, path1=s_file6)


                    if(result < THRESHOLD):
                        fail_result = tk.Label(root_v, text="{:.2f}".format(result-5)+"% Match. Not Verified!", font=('calibre',13,'bold'), bg='red')
                        fail_result.place(x=210, y=520)
                        messagebox.showerror("Failure: Signatures Do Not Match",
                                            "Signatures are "+str(result-5)+f" % similar!!")
                        fail_result.destroy()
                    else:
                        valid_result = tk.Label(root_v, text="{:.2f}".format(result+5)+"% Match. Verified!", font=('calibre',13,'bold'), bg='green')
                        valid_result.place(x=210, y=520)
                        messagebox.showinfo("Success: Signatures Match",
                                            "Signatures are "+str(result+5)+f" % similar!!")
                        valid_result.destroy()

                    cv2.destroyAllWindows()

                    dataset.destroy()
                    d_result.destroy()

                    return True


        def gui_destroy():
            root_v.destroy()



        self.uname_label = tk.Label(root_v, text="Verify User Signature", font=('calibre',14,'bold'), bg='medium aquamarine')
        self.uname_label.place(x=140, y=25)

        # creating a label for name using widget Label
        self.name_label = tk.Label(root_v, text = 'Username:', font=('calibre',10,'bold'))
        self.name_label.place(x=60, y=90)

        # creating a entry for input name using widget Entry
        self.name_entry = tk.Entry(root_v, bd=3, font=('calibre',10,'normal'))
        self.name_entry.place(x=170, y=90)

        # creating a button using the widget button that will call the submit function
        self.sub_btn = tk.Button(root_v, text = 'Check Data', font=('calibre',10,'normal'), command = check_data)
        self.sub_btn.place(x=340, y=85)


        # Upload
        self.img_message = tk.Label(root_v, text="New Signature:", font=('calibre',10,'bold'))
        self.img_message.place(x=60, y=140)
        # Image Submit
        self.image_path_entry1 = tk.Entry(root_v, bd=3, font=('calibre',10,'normal'))
        self.image_path_entry1.place(x=170, y=140)
        # Browse Button
        self.img_browse_button = tk.Button(
            root_v, text="Browse", font=('calibre',10,'normal'), command=lambda: browsefunc(ent=self.image_path_entry1))
        self.img_browse_button.place(x=340, y=135)



        # Verify Button
        self.verify_button = tk.Button(
            root_v, text="Verify", font=('calibre',12,'bold'), bg='gold2', command=lambda: checkSimilarity(window=root_v,
                                                                        path0=self.name_entry.get(),
                                                                        path1=self.image_path_entry1.get(),), width=8)
        self.verify_button.place(x=198, y=190)


        # Exit Button
        self.go_exit = tk.Button(
            root_v, text="Exit", bg='tomato', font=('calibre',12,'bold'), command=lambda: gui_destroy(), width=5)
        self.go_exit.place(x=215, y=240)


        # performing an infinite loop for the window to display
        root_v.mainloop()



root = tk.Tk()
root.configure(bg='wheat1')
root.geometry("500x300+50+50")

app = Application(master=root)
app.master.title("Signature Registration & Verification System")
app.mainloop()
