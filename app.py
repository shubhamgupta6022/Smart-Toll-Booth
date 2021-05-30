# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
from flask import Flask, render_template, request, redirect, url_for, session, g, jsonify
from flask_mysqldb import MySQL
from flask_mail import Mail
from datetime import timedelta
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy  import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
import MySQLdb.cursors
import json
import re
import os
import stripe
from datetime import datetime
current_datetime = datetime.now()



# ### Part 1: Extract license plate from sample image
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)



wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

test_image_path = "Plate_examples/india_car_plate.jpg"
vehicle, LpImg,cor = get_plate(test_image_path)


# ## Part 2: Segementing license characters



if (len(LpImg)): #check if there is at least one license image
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

   
# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 30, 60

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 1<=ratio<=3.5: # Only select contour with defined ratio
        if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

            # Sperate number and gibe prediction
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)

print("Detect {} letters...".format(len(crop_characters)))


# ## Load pre-trained MobileNets model and predict



# Load model architecture, weight and labels
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

cols = len(crop_characters)

final_string = ''
for i,character in enumerate(crop_characters):
    title = np.array2string(predict_from_model(character,model,labels))
    final_string+=title.strip("'[]")

print(final_string)


 

app = Flask(__name__)

app.config['STRIPE_PUBLIC_KEY']='pk_test_RFPRlEBabMK30OFPGfD79CYB00ptz4nZyB'
app.config['STRIPE_SECRET_KEY']='sk_test_VhzhwICHa2qwbLkE9Lj6dqCZ00nubLxqEX'


stripe.api_key = app.config['STRIPE_SECRET_KEY']

# Set your secret key. Remember to switch to your live secret key in production!
# See your keys here: 
stripe.api_key = 'sk_test_VhzhwICHa2qwbLkE9Lj6dqCZ00nubLxqEX'

customer = stripe.Customer.create(
  name='Jenny Rosen',
  address={
    'line1': '510 Townsend St',
    'postal_code': '98140',
    'city': 'San Francisco',
    'state': 'CA',
    'country': 'US',
  },
)



# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1234'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'smart_toll_booth'

# Intialize MySQL
mysql = MySQL(app)
bootstrap = Bootstrap(app)



@app.route('/', methods = ['GET', 'POST'])
def mainpage():
    if(request.method=='POST'):
        username = request.form['username_login']
        password = request.form['password']
        
        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute('INSERT INTO history VALUES (NULL,%s,%s, %s, %s, %s, %s, %s)', (final_string, current_datetime, 'Rajpura', 'Single','Car/Jeep/Van','35','No', ))
        curl.execute("SELECT date_time,toll_name,type_of_journey,fee,paid FROM history WHERE vehicle_no=%s ORDER BY date_time DESC",(username,))
        data  = curl.fetchall()
        mysql.connection.commit()
        #curl.close() 

        if data:
            curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            curl.execute("SELECT date_time,toll_name,type_of_journey,fee,paid FROM history WHERE vehicle_no=%s ORDER BY date_time DESC",(username,))
            data  = curl.fetchall()
            curl.execute("SELECT SUM(fee) Bill FROM history WHERE vehicle_no=%s AND paid='No'",(username,))
            amount=curl.fetchall()
            return render_template('dashboard.html', name=username, output_data =data, amount=amount,key=app.config['STRIPE_PUBLIC_KEY'])
        else:
            return 'Error! wrong ID or password'

    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        username = request.form['username_login']
        password = request.form['password']
        
        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute('INSERT INTO history VALUES (NULL,%s,%s, %s, %s, %s, %s, %s)', (final_string, current_datetime, 'Rajpura', 'Single','Car/Jeep/Van','35','No', ))

        curl.execute("SELECT * FROM login WHERE username_login=%s AND password = %s",(username, password,))
        account  = curl.fetchone()
        mysql.connection.commit()


        if account:
            curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            curl.execute("SELECT date_time,toll_name,type_of_journey,fee,paid FROM history WHERE vehicle_no=%s ORDER BY date_time DESC",(username,))
            data  = curl.fetchall()
            curl.execute("SELECT SUM(fee) Bill FROM history WHERE vehicle_no=%s AND paid='No'",(username,))
            amount=curl.fetchall()
            return render_template('dashboard.html', name=username, output_data =data, amount=amount,key=app.config['STRIPE_PUBLIC_KEY'])
        else:
            return 'Error! wrong ID or password'
                
    return render_template('login.html')


@app.route('/test')
def test():
    return render_template('test.html',final_string=final_string)

@app.route('/profile')
def profile():
    username = 'PC1313'

    curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    curl.execute("SELECT date_time,toll_name,type_of_journey,fee,paid FROM history WHERE vehicle_no=%s ORDER BY date_time DESC",(username,))
    data  = curl.fetchall()
    curl.execute("SELECT SUM(fee) Bill FROM history WHERE vehicle_no=%s AND paid='No'",(username,))
    amount=curl.fetchall()
    return render_template('profile.html', output_data=data, name=username, amount=amount,key=app.config['STRIPE_PUBLIC_KEY'])


@app.route('/checkout', methods=['POST'])
def checkout():
    amount = 500

    customer = stripe.Customer.create(
        email='sample@customer.com',
        source=request.form['stripeToken']
    )

    stripe.Charge.create(
        customer=customer.id,
        amount=amount,
        currency='usd',
        description='Flask Charge'
    )

    return render_template('checkout.html', amount=amount)

@app.route('/dashboard')
@login_required
def dashboard():
    vehicle_no='PC1313'
    #curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor) date_time,toll_name,type_of_journey,type_of_vehicle,fee
    curl.execute("SELECT date_time,toll_name,type_of_journey,type_of_vehicle,fee FROM history WHERE vehicle_no=%s",(vehicle_no,))
    data  = curl.fetchall()

    return render_template('dashboard.html', output_data =data, name=current_user.username )

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/contact", methods = ['GET', 'POST'])
def contact():
    if request.method=='POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        #cursor.execute('INSERT INTO contact VALUES (NULL, %s, %s)', (vehicle_no, email_id,))
        cursor.execute('INSERT INTO contact VALUES (NULL, %s, %s, %s, %s)', (name, email, subject,message, ))
        mysql.connection.commit()
     
    return render_template('contact.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'name' in request.form and 'email_id' in request.form and 'vehicle_no' in request.form:
        # Create variables for easy access
        name = request.form['name']
        email_id = request.form['email_id']
        vehicle_no = request.form['vehicle_no']
        gender = request.form['gender']
        street_address = request.form['street_address']
        city = request.form['city']
        state = request.form['state']
        pin_code = request.form['pin_code']
        phone_no = request.form['phone_no']


        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM login WHERE username_login = %s', (vehicle_no,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO register VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (name, email_id, vehicle_no, gender, street_address, city, state , pin_code, phone_no, ))
            cursor.execute('INSERT INTO login VALUES (NULL, %s, %s, %s, %s)', (vehicle_no, email_id, name, phone_no, ))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            return redirect('/login')

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)


@app.route("/pricing")
def pricing():
    return render_template('pricing.html')

app.run(debug=True)