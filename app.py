from flask import Flask , render_template , request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))



app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")
    
   
@app.route("/",methods=["POST"])
def predict():
    #HTML > PY
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)

    #.PY > HTML
    return render_template('sub.html', data=pred)

    #.PY > HTML
    #return render_template("sub.html", n = name )    





if __name__=="__main__":
    app.run(debug=True)    