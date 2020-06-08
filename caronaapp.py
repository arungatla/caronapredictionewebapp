from flask import Flask,render_template,request

import pickle

app=Flask(__name__)

file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()


@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method=="POST":
        mydict=request.form
        
        fever=int(mydict['fever'])
        age=int(mydict['age'])
        pain=int(mydict['pain'])
        diffBreath=int(mydict['diffBreath'])
        runnyNose=int(mydict['runnyNose'])
        input_features=[fever,pain,age,runnyNose,diffBreath]
        infprob=clf.predict_proba([input_features])
        infprob=infprob[0][1]
        print(infprob)
        return render_template('show.html',inf=round(infprob*100))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
    