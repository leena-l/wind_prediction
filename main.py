from flask import Flask,jsonify,render_template,request
import pickle


# from keras import models
file=open('my_model.pkl','rb')
model=pickle.load(file)

#file.close()

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':
        mydict=request.form
        gph=mydict['gph']
        wdir=mydict['wdir']
        month=mydict['month']
        day=mydict['day']
        reltime=mydict['reltime']
        mwspeed=mydict['mwspeed']
        npv=mydict['npv']
        npv1=mydict['npv1']


        
        
        input_feature=[gph,wdir,month,day,reltime,mwspeed,npv,npv1]
        #input_feature=[100,1,45,1,1,0]

        #input_feature=[1,-1,0,1,-1,0,0,1,0]
        infprob=model.predict([input_feature])
        
        #infprob = infprob*100
        return render_template('result.html',inf=infprob)
   
    return render_template('index.html')
   
if __name__ == '__main__'  :
    app.run(debug=False) 
