# Signature-Validator
Application to detect the similarity of two signatures.

This Application helps mathematically evaluate similarity of two signatures using GABOR Transform. 

# Update (over everything written below)
Run 'gui_sign_app.py' which will open a GUI. It controlls the signature uploader and validator both. The validation result is also displayed on the GUI.

## Dataset
First Create a dataset of Signatures in assets folder. Create folder with user names, such as: zayed, raihan, shamim, partho, etc. Then in each folder, store the signature images in ".jpg" or any format (Then change the format in code in respective places).
The store images can have names such as: zayed1.jpg, zayed2.jpg, zayed3.jpg, ... etc. Create as many users as you want.


## Validation
After running "validator.py", in the tkinter UI:
Give the name of the user you want to validate.
You can display the user database by clicking view.
Then you can upload or capture the signature which you want to validate.
After clicking validate, the popup will show the percentage match of the signatures.
The signatures are compared using GABOR Transform Method.


### Prerequisites
1. tkinter
2. OpenCV
3. Scipy
4. Scikit-learn
5. Scikit-image
6. numpy
7. math
8. matplotlib


### Run
1. `python validator.py`


### Preview

https://user-images.githubusercontent.com/30951078/152105000-f5cc8d96-7c4f-4625-bf8f-3fae407146b7.mp4



### Please open an issue if
* you have any suggestion to improve this project
* you noticed any problem or error
