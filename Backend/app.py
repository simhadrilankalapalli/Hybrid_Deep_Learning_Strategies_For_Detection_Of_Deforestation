from flask import Flask, request, render_template, send_from_directory,session,flash
import pandas as pd
import string
import os
import mysql.connector
import numpy as np
from datetime import timedelta
import sys
from PIL import Image
import base64
import io
import re

import PIL.Image
from datetime import datetime
app = Flask(__name__)
app.config['SECRET_KEY'] = 'the random string'

classes = ['System Detected as deforestation',
            'System Detected as forest']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/reg")

def reg():
    return render_template("ureg.html")
@app.route('/regback',methods = ["POST"])
def regback():
    if request.method=='POST':
        name=request.form['name']
        email=request.form['email']
        pwd=request.form['pwd']
        cpwd=request.form['cpwd']
        pno=request.form['pno']



    #email = request.form["email"]

        print("**************")
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="deforestation"
        )
        mycursor = mydb.cursor()
        print("**************")
        sql = "select * from ureg"
        result = pd.read_sql_query(sql, mydb)
        email1 = result['email'].values
        print(email1)
        if email in email1:
            flash("email already exists","warning")
            return render_template('ureg.html')
        if(pwd==cpwd):
            sql = "INSERT INTO ureg (name,email,pwd,pno) VALUES(%s,%s,%s,%s)"
            val = (name, email, pwd, pno)
            mycursor.execute(sql, val)
            mydb.commit()
            flash("You registered successfully", "success")

            return render_template('user.html')
        else:
            flash("Password and Confirm Password are not same", "danger")
            return render_template('ureg.html')
    flash("Something wrong", "danger")
    return render_template('user.html')


@app.route('/userlog',methods=['POST', 'GET'])
def userlog():
    global name, name1
    global user
    if request.method == "POST":

        username = request.form['email']
        password1 = request.form['pwd']
        print('p')
        mydb = mysql.connector.connect(host="localhost", user="root", passwd="", database="deforestation")
        cursor = mydb.cursor()
        sql = "select * from ureg where email='%s' and pwd='%s'" % (username, password1)
        print('q')
        x = cursor.execute(sql)
        print(x)
        results = cursor.fetchall()
        print(results)
        if len(results) > 0:
            print('r')
            # session['user'] = username
            # session['id'] = results[0][0]
            # print(id)
            # print(session['id'])
            flash("Welcome to website", "success")
            return render_template('userhome.html', msg=results[0][1])
        else:
            flash("Invalid Email/password", "danger")
            return render_template('user.html')

    return render_template('user.html')
@app.route("/userhome")
def userhome():
    return render_template("userhome.html")

@app.route("/upload", methods=["POST","GET"])
def upload():
    print('a')

    myfile=request.files['file']
    fn=myfile.filename
    mypath=os.path.join('Z:\INDIRA\Indu\Backup codes\Identifying deforestation using AI enabled satellite Image processing\CODE\images', fn)
    myfile.save(mypath)

    print("{} is the file name",fn)
    print ("Accept incoming file:", fn)
    print ("Save it to:", mypath)
    #import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import load_model
    img=r"Z:\INDIRA\Indu\Backup codes\Identifying deforestation using AI enabled satellite Image processing\CODE\dataset\test\deforestation\3.jpg"
    new_model = load_model(r"Z:\INDIRA\Indu\Backup codes\Identifying deforestation using AI enabled satellite Image processing\CODE\CNN_1.h5")
    test_image = image.load_img(mypath, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = new_model.predict(test_image)

    prediction=classes[np.argmax(result)]
    return render_template("template.html",image_name=fn, text=prediction)

@app.route('/upload/<filename>')
def send_image(filename):
    print("hello world")
    return send_from_directory(r"Z:\INDIRA\Indu\Backup codes\Identifying deforestation using AI enabled satellite Image processing\CODE\images", filename)
@app.route('/upload1')
def upload1():
    return render_template("upload.html")

@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)