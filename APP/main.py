from flask import Flask,url_for,send_file ,render_template,request,redirect,send_from_directory,url_for,session
import os
import tensorflow as  tf
import numpy as np
from tensorflow import keras
from PIL import Image 
from skimage.io import imshow
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2lab,lab2rgb


app=Flask(__name__)
app.secret_key='ahbcdefghqwerty'
img_type=["PNG","JPG","JPEG","GIF"]

app_root=os.path.dirname(os.path.abspath(__file__))



def ExtractTestInput(ipath):
    img = cv2.imread(ipath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_ = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2Lab)
    img_=img_.astype(np.float32)
    img_lab_rs = cv2.resize(img_, (224, 224)) # resize image to network input size
    img_l = img_lab_rs[:,:,0] # pull out L channel
    #img_l -= 50
    img_l_reshaped = img_l.reshape(1,224,224,1)
    
    return img_l_reshaped

def allowed(filename):
    if not "." in filename:
        return False
    ext=filename.rsplit(".",1)[1]

    if ext.upper() in img_type :
        return True
    else :
        return False       

@app.route("/",methods=["GET","POST"])
def main():
    if request.method=="POST":
        if request.files:
            image=request.files["image"]
            if image.filename == "":
                print("image must have a name ")
                return redirect(request.url)
            
            if not allowed(image.filename):
                print("image extension in not allowed")
                return redirect(request.url)
            target=os.path.join(app_root,'static/')
            target1=os.path.join(app_root)
            filenamed=image.filename
            c=filenamed
            c1="op"+c[1:]
            print(c1)
            session["c1"]=c1
            destination="/".join([target,filenamed])
            image.save(destination)
            print("image saved")
        
            z='app\\static\\'
            z1='app\\static\\'
            full_path = os.path.join(z, filenamed)
            full_path1 = os.path.join(z1, c1)
            print(full_path)
            print(full_path1)
            model=keras.models.load_model('app\\imgcol.h5')
            ImagePath= full_path

            img1_color = []
            img1 = img_to_array(load_img(ImagePath))
            h=img1.shape[0]
            w=img1.shape[1]
            img1 = resize(img1, (256, 256))
            img1_color.append(img1)
            img1_color = np.array(img1_color, dtype=float)
            img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
            img1_color = img1_color.reshape(img1_color.shape+(1,))
            output1 = model.predict(img1_color)
            output1 = output1*128
            result = np.zeros((256, 256, 3))
            result[:,:,0] = img1_color[0][:,:,0]
            result[:,:,1:] = output1[0]
            result=resize(result,(h,w))
            io.imshow(lab2rgb(result))
            io.imsave(full_path1,lab2rgb(result))

            
            return render_template("op.html",a=c,b=c1)
    return render_template("home.html")    


@app.route('/download_file',methods=["GET","POST"])
def download_file():
    a=session.get("c1",None)
    return send_from_directory("static",a,as_attachment=True)
    
    
    
    
if __name__=="__main__":
    app.run(debug=True) 
