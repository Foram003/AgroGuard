from flask import Flask, render_template, request, jsonify,flash,redirect,url_for,session
from markupsafe import Markup
import pandas as pd
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pickle
import requests
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import mysql.connector
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash


# Load models and encoders
classifier = load_model('Trained_model.h5')
crop_recommendation_model = pickle.load(open('Crop_Recommendation.pkl', 'rb'))
fertilizer_rec_model = pickle.load(open('classifier.pkl', 'rb'))
fruit_quality_model = load_model('fruit_quality_inception_model.h5', compile=False)

# Load encoders
encode_soil = pickle.load(open('soil_encoder.pkl', 'rb'))
encode_crop = pickle.load(open('crop_encoder.pkl', 'rb'))

app = Flask(__name__)
WEATHER_API_KEY = "9df0601d3825223d76ad73e5eaa53615"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Weather and location functions
def get_current_location():
    response = requests.get("http://ip-api.com/json/")
    data = response.json()
    return data['lat'], data['lon']

def fetch_weather_data(lat, lon):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    response = requests.get(WEATHER_API_URL, params=params)
    data = response.json()
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    rainfall = data.get('rain', {}).get('1h', 0)
    return temperature, humidity, rainfall

# Pest prediction function
def pred_pest(pest):
    try:
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0
        result = np.argmax(classifier.predict(test_image), axis=1)
        return result
    except Exception as e:
        print(f"Image Processing Error: {e}")
        return 'x'

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")
app.secret_key = os.urandom(24)
db_config = {
    'host': 'localhost',
    'user': 'root',  # Default MySQL user for XAMPP/MAMP
    'password': '',  # Default password for XAMPP/MAMP (leave empty if no password)
    'database': 'agroguard'  # Database name you created in phpMyAdmin
}

@app.route("/sign_in",methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        username = request.form['username']
        raw_password = request.form['password']

        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], raw_password):
            # Store user info in session
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('crop'))  # Redirect to crop recommendation or dashboard
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('sign_in'))

    return render_template("SignIn.html")

@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        # Collect form data
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        raw_password = request.form['password']
        hashed_password = generate_password_hash(raw_password)

        # Connect to MySQL database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Check if the email or username already exists
        cursor.execute("SELECT * FROM users WHERE email = %s OR username = %s", (email, username))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Error: Email or Username already exists.')
            return redirect(url_for('index'))

        # Insert new user into the database
        cursor.execute("INSERT INTO users (fullname, email, username, password) VALUES (%s, %s, %s, %s)",
                       (fullname, email, username, hashed_password))
        conn.commit()
        conn.close()

        # Show a success message and redirect to the sign-in page
        flash('Account created successfully!')
        return redirect(url_for('index'))  #Assuming you have a 'sign_in' route

    # If the request method is GET, simply render the signup page
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()  #This clears all session data
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))  #Or redirect to 'sign_in' if preferred

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/CommunityPlatform.html")
def community():
    return redirect(url_for('community_platform'))


@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

#Community platform routes
community_posts = [
    {
        "name": "Farmer A",
        "message": "How's the crop yield this season?",
        "timestamp": "2025-02-04 10:00",
        "media": None,
        "media_type": None,
        "reactions": {"like": 0, "love": 0, "haha": 0},
        "comments": []
    }
]

@app.route("/Community")
def community_platform():
    if 'user_id' not in session:
        return redirect(url_for('sign_in'))

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT p.*, u.username
        FROM posts p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.timestamp DESC
    """)
    posts = cursor.fetchall()

    for post in posts:
        post_id = post["id"]

        # Fetch reactions (group by type and count)
        cursor.execute("""
            SELECT reaction_type, COUNT(*) as count
            FROM reactions
            WHERE post_id = %s
            GROUP BY reaction_type
        """, (post_id,))
        reactions = cursor.fetchall()
        post["reactions"] = {r["reaction_type"]: r["count"] for r in reactions}

        # Fetch comments
        cursor.execute("""
            SELECT c.text, u.username
            FROM comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.post_id = %s
            ORDER BY c.timestamp ASC
        """, (post_id,))
        comments = cursor.fetchall()
    

        # ✅ Use nested comment structure
        post["comments"] = get_comments_nested(post_id)

    conn.close()
    return render_template("CommunityPlatform.html", posts=posts, username=session['username'])

import os
from werkzeug.utils import secure_filename
from flask import flash

@app.route("/communitymessage", methods=["POST"])
def community_message():
    if 'user_id' not in session:
        return redirect(url_for('sign_in'))

    user_id = session['user_id']
    message = request.form.get("message")
    media = request.files.get("media")

    media_path = None
    media_type = None

    if media and media.filename != '':
        filename = secure_filename(media.filename)
        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        media.save(file_path)

        media_path = file_path
        media_type = media.content_type.split("/")[0]  

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO posts (user_id, message, media_path, media_type)
            VALUES (%s, %s, %s, %s)
        """, (user_id, message, media_path, media_type))
        conn.commit()
        conn.close()
        flash("Post created successfully!")
    except Exception as e:
        flash(f"Error creating post: {str(e)}")

    return redirect(url_for("community_platform"))


@app.route("/react_to_post/<int:post_id>", methods=["POST"])
def react_to_post(post_id):
    if 'user_id' not in session:
        return jsonify(success=False, message="Login required.")

    user_id = session["user_id"]
    reaction_type = request.form.get("reaction")

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Check if the user has already reacted with this reaction type
    cursor.execute("""
        SELECT id FROM reactions WHERE post_id = %s AND user_id = %s AND reaction_type = %s
    """, (post_id, user_id, reaction_type))
    existing = cursor.fetchone()

    if existing:
        # Remove the reaction (toggle off)
        cursor.execute("""
            DELETE FROM reactions WHERE id = %s
        """, (existing[0],))
    else:
        # Add a new reaction
        cursor.execute("""
            INSERT INTO reactions (post_id, user_id, reaction_type) VALUES (%s, %s, %s)
        """, (post_id, user_id, reaction_type))

    # Now, fetch updated reaction counts for this post
    cursor.execute("""
        SELECT reaction_type, COUNT(*) FROM reactions WHERE post_id = %s GROUP BY reaction_type
    """, (post_id,))
    reactions = cursor.fetchall()

    # Construct a dictionary of reactions count
    reaction_counts = {
        'like': 0,
        'love': 0,
        'haha': 0
    }
    for reaction in reactions:
        reaction_counts[reaction[0]] = reaction[1]

    conn.commit()
    conn.close()

    # Return success along with updated reaction counts
    return jsonify(success=True, reactions=reaction_counts)

@app.route("/comment_on_post/<int:post_id>", methods=["POST"])
def comment_on_post(post_id):
    if 'user_id' not in session:
        return jsonify(success=False, message="Login required.")

    user_id = session["user_id"]
    comment_text = request.form.get("comment")
    parent_comment_id = request.form.get("parent_comment_id")  # Optional

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    if parent_comment_id:
        cursor.execute("""
            INSERT INTO comments (post_id, user_id, text, parent_comment_id)
            VALUES (%s, %s, %s, %s)
        """, (post_id, user_id, comment_text, parent_comment_id))
    else:
        cursor.execute("""
            INSERT INTO comments (post_id, user_id, text)
            VALUES (%s, %s, %s)
        """, (post_id, user_id, comment_text))
    
    conn.commit()
    # Get user name (optional, but needed to display in frontend)
    cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
    user_name = cursor.fetchone()[0]

    

    #return jsonify(success=True, comment={"name": user_name, "text": comment_text})
    conn.close()
    
    #return jsonify(success=True)
    
    return redirect(url_for("community_platform"))



def get_comments_nested(post_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT c.*, u.username 
        FROM comments c
        JOIN users u ON c.user_id = u.id
        WHERE c.post_id = %s 
        ORDER BY c.timestamp ASC
    """, (post_id,))
    rows = cursor.fetchall()
    conn.close()

    comment_map = {}
    top_level_comments = []

    for row in rows:
        row['replies'] = []
        comment_map[row['id']] = row

    for row in rows:
        if row['parent_comment_id']:
            parent = comment_map.get(row['parent_comment_id'])
            if parent:
                parent['replies'].append(row)
        else:
            top_level_comments.append(row)

    return top_level_comments

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part in request."

        file = request.files['image']
        if file.filename == '':
            return "No selected file."

        file_path = os.path.join('static/user uploaded', file.filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)
        if pred == 'x':
            return render_template('unaptfile.html')

        pest_classes = [
            'aphids', 'armyworm', 'beetle', 'bollworm', 'earthworm',
            'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem borer'
        ]

        if 0 <= pred[0] < len(pest_classes):
            pest_identified = pest_classes[pred[0]]
        else:
            return "Invalid model prediction."

        return render_template(f"{pest_identified}.html", pred=pest_identified)

@app.route('/fertilizer_recommend', methods=['POST'])
def fertilizer_recommend():
    if request.method == 'POST':
        try:
            lat, lon = get_current_location()
            Temperature, Humidity = fetch_weather_data(lat, lon)[:2]  
            Soil_Type = request.form['soil_type']
            Crop_Type = request.form['crop_type']
            Nitrogen = int(request.form['nitrogen'])
            Phosphorous = int(request.form['phosphorous'])
            Potassium = int(request.form['potassium'])
            Soil_Type_encoded = encode_soil.transform([Soil_Type])[0]
            Crop_Type_encoded = encode_crop.transform([Crop_Type])[0]
            fertilizer_encoder = pickle.load(open('fertilizer_encoder.pkl', 'rb'))
            data = np.array([[Temperature, Humidity, Soil_Type_encoded, Crop_Type_encoded,
                              Nitrogen, Potassium, Phosphorous]])
            prediction_encoded = fertilizer_rec_model.predict(data)[0]
            final_prediction = fertilizer_encoder.inverse_transform([prediction_encoded])[0]
            print(f"Input data: {data}")
            print(f"Encoded prediction: {prediction_encoded}")
            print(f"Final prediction: {final_prediction}")
            return render_template(
                'Fertilizer-Result.html',
                prediction=final_prediction,
                pred='img/img_fertlizer/' + final_prediction + '.jpg',
                temperature=Temperature,
                humidity=Humidity,
            )
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error during prediction: {e}"

# -------------------- Fruit Quality Assessment --------------------
class_labels = ["A", "B", "C", "D"]
def predict_fruit_quality(image_path):
    try:
        print(f"Loading image from: {image_path}")
        img = image.load_img(image_path, target_size=(299, 299))
        print("Image loaded successfully.")
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
        print("Image preprocessed.")
        prediction = fruit_quality_model.predict(img_array)
        print(f"Model prediction: {prediction}")
        return class_labels[np.argmax(prediction)]
    except Exception as e:
        print(f"Error in predict_fruit_quality: {e}")
        raise e

@app.route("/quality_assesment", methods=["GET", "POST"])
def quality_assesment():
    if request.method == "GET":
        return render_template("quality_assesment.html")

    if 'image' not in request.files:
        print("No file uploaded.")
        return render_template("quality_assesment.html", error="No file uploaded.")

    file = request.files['image']
    if file.filename == '':
        print("No selected file.")
        return render_template("quality_assesment.html", error="No selected file.")

    file_path = os.path.join('uploads', file.filename)  
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  
    print(f"Saving file to: {full_path}")
    file.save(full_path)
    try:
        quality_result = predict_fruit_quality(full_path)
        print(f"Prediction result: {quality_result}")
        print("Image path:", file_path)
        print("Prediction:", quality_result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template("quality_assesment.html", error="An error occurred during prediction.")

    return render_template("quality_assesment.html", image=file_path, prediction=quality_result)

import pandas as pd
from sklearn.model_selection import train_test_split
crop = pd.read_csv('Data/crop_recommendation_new.csv')
X = crop.iloc[:, :-1].values  
Y = crop.iloc[:, -1].values  

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
crop_recommendation_model = pickle.load(open('Crop_Recommendation.pkl', 'rb'))

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,  
    feature_names=['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
    class_names=np.unique(Y),  
    mode="classification"
)

@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        try:
            lat = float(request.form['latitude'])
            lon = float(request.form['longitude'])
            temperature, humidity, rainfall = fetch_weather_data(lat, lon)
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['potassium'])
            ph = float(request.form['ph'])
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            final_prediction = crop_recommendation_model.predict(data)[0]
            pred_probs = crop_recommendation_model.predict_proba(data)
            top_class = np.argmax(pred_probs)
            explanation = explainer.explain_instance(
                data[0], 
                crop_recommendation_model.predict_proba,labels=[top_class]
            )
            lime_html = explanation.as_html()
            styled_lime_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>LIME Explanation</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #f8f9fa;
                        text-align: center;
                        margin: 20px;
                    }}
                    .container {{
                        max-width: 900px;
                        margin: auto;
                        background: white;
                        padding: 20px;
                        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
                        border-radius: 10px;
                    }}
                    h1 {{
                        color: #333;
                    }}
                    iframe {{
                        width: 100%;
                        height: 600px;
                        border: none;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Model Explanation with LIME</h1>
                    {lime_html}
                </div>
            </body>
            </html>
            """ 
            lime_html_path = "static/lime_explanation.html"
            with open(lime_html_path, 'w', encoding='utf-8') as f:
                f.write(styled_lime_html)

            return render_template(
                'crop-result.html',
                prediction=final_prediction,
                pred='img/crop/' + final_prediction + '.jpg',  
                temperature=temperature,
                humidity=humidity,
                rainfall=rainfall,
                lime_explanation=lime_html_path
            )
        except Exception as e:
            return f"Error during prediction: {e}"
        
if __name__ == '__main__':
    app.run(debug=True)