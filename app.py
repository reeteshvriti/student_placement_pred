from  flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('svc_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict_placement():
    cgpa = float(request.form.get('cgpa'))
    com = int(request.form.get('com'))
    con = int(request.form.get('con'))

    # prediction
    result = model.predict(np.array([cgpa,com, con]).reshape(1,3))

    if result[0] == 1:
        result = 'Student has more chances of getting placed'
    else:
        result = 'Student has less chance of getting placed and requires training'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)


'''
if __name__ == "__main__":
    app.run(debug=True)
'''


