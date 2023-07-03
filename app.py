from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("main.html")

@app.route("/login.html")
def login():
    return render_template("login.html")

@app.route("/survey.html")
def survey():
    return render_template("survey.html")

@app.route("/autm6.html")
def autm6():
    return render_template("autm6.html")

@app.route("/autm9.html")
def autm9():
    return render_template("autm9.html")

@app.route("/asq.html",methods=['GET', 'POST'])
def auts():
    return render_template("asq.html")

@app.route("/autp.html")
def autp():
    return render_template("autp.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/calcm.html",methods=['GET', 'POST'])
def calcm():
    return render_template("calcm.html")


@app.route("/home.html",methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route("/calc.html")
def calc():
    return render_template("calc.html")

@app.route("/surveyd.html")
def surveyd():
    return render_template("surveyd.html")

@app.route("/dysmain.html",methods=['GET', 'POST'])
def dysmain():
    return render_template("dysmain.html")

@app.route("/resultsd.html")
def resultsd():
    return render_template("resultsd.html")

# @app.route("/predictiond.html")
# def predictiond():
#     return render_template("predictiond.html")



@app.route("/6-8d.html")
def d6():
    return render_template("6-8d.html")


@app.route("/6-8d1.html")
def d1_68():
    return render_template("6-8d1.html")


@app.route("/6-8d2.html")
def d2_68():
    return render_template("6-8d2.html")


@app.route("/6-8d3.html")
def d3_68():
    return render_template("6-8d3.html")

@app.route("/6-8d4.html")
def d4_68():
    return render_template("6-8d4.html")

@app.route("/9-11d.html")
def d9():
    return render_template("9-11d.html")


@app.route("/9-11d1.html")
def d1_911():
    return render_template("9-11d1.html")


@app.route("/9-11d2.html")
def d2_911():
    return render_template("9-11d2.html")


@app.route("/9-11d3.html")
def d3_911():
    return render_template("9-11d3.html")

@app.route("/9-11d4.html")
def d4_911():
    return render_template("9-11d4.html")

@app.route("/6-8.html")
def main6():
    return render_template("6-8.html")


@app.route("/6-8g1.html")
def g1_68():
    return render_template("6-8g1.html")


@app.route("/6-8g2.html")
def g2_68():
    return render_template("6-8g2.html")


@app.route("/6-8g3.html")
def g3_68():
    return render_template("6-8g3.html")


@app.route("/6-8g4.html")
def g4_68():
    return render_template("6-8g4.html")


@app.route("/9-11.html")
def main9():
    return render_template("9-11.html")


@app.route("/9-11g1.html")
def g1_911():
    return render_template("9-11g1.html")


@app.route("/9-11g2.html")
def g2_911():
    return render_template("9-11g2.html")


@app.route("/9-11g3.html")
def g3_911():
    return render_template("9-11g3.html")


@app.route("/9-11g4.html")
def g4_911():
    return render_template("9-11g4.html")


@app.route('/results', methods=['GET', 'POST'])
def results():
    return render_template("results.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Get the scores from the quiz
    lang_vocab = float(request.form['lang_vocab'])
    memory_score =float(request.form['memory_score'])
    speed_score = float(request.form['speed_score'])
    visual_score = float(request.form['visual_score'])
    audio_score = float(request.form['audio_score'])
    survey_score = float(request.form['survey_score'])

    # label 0 - low
    # label 1 - Moderate
    # label 2 - High
    features = ['Language_vocab', 'Memory', 'Speed', 'Visual_discrimination', 'Audio_Discrimination', 'Survey_Score']
    file = pd.read_csv('labelled_dysx.csv')
    X = file[features]
    y = file['Label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = Sequential()
    model.add(Dense(12, input_shape=(6,), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
    training = model.fit(x_train, y_train, epochs=20, verbose=True)
    pred = model.predict([x_test]).round()

    #2D numpy array created with the values input by the user.
    array = np.array([[lang_vocab, memory_score, speed_score, visual_score, audio_score, survey_score]])
    #The output given by model is converted into an int and stored in label.
    label=int(model.predict(array))
    print("label ",label)
    # Return the predicted performance level
    return render_template('prediction.html', lang_vocab=lang_vocab, speed_score=speed_score, memory_score=memory_score, visual_score=visual_score, audio_score=audio_score, survey_score=survey_score, result=label)

@app.route('/predictd', methods=['GET', 'POST'])
def predictd():
    # Get the scores from the quiz
    seq_pattern = float(request.form['seq'])
    memory_score =float(request.form['mem'])
    speed_score = float(request.form['speed'])
    percept_score = float(request.form['perc'])
    abs_score = float(request.form['abs'])
    surveyd_score = float(request.form['surveyd'])

    # label 0 - low
    # label 1 - Moderate
    # label 2 - High
    features = ['Sequence_pattern', 'Memory', 'Speed' ,'Perception' , 'Abstract_Reasoning', 'Surveyd_Score']
    file = pd.read_csv('labelled_dyscal.csv')
    X = file[features]
    y = file['Label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = Sequential()
    model.add(Dense(12, input_shape=(6,), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
    training = model.fit(x_train, y_train, epochs=20, verbose=True)
    pred = model.predict([x_test]).round()

    #2D numpy array created with the values input by the user.
    array = np.array([[seq_pattern, memory_score, speed_score, percept_score, abs_score, surveyd_score]])
    #The output given by model is converted into an int and stored in label.
    label=int(model.predict(array))
    print("label ",label)
    # Return the predicted performance level
    return render_template('predictiond.html',result=label)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
