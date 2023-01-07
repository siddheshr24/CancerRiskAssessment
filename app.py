import joblib
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from flask import Flask,render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
#model1 = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)
app = Flask(__name__, static_url_path = "", static_folder = "static")
bootstrap = Bootstrap(app)



# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'cancer_assessment'

# Intialize MySQL
mysql = MySQL(app)


@app.route('/cancer_assessment/withoutlogin')
def withoutlogin():
    return render_template('withoutlogin.html')



@app.route('/cancer_assessment/BLOG')
def BLOG():
    return render_template('BLOG.html')


@app.route('/cancer_assessment/liventry')
def liventry():
    return render_template('liventry.html')

@app.route('/cancer_assessment/pancentry')
def pancentry():
    return render_template('pancentry.html')

@app.route('/cancer_assessment/panc')
def panc():
    return render_template('panc.html')

@app.route('/cancer_assessment/liv')
def liv():
    return render_template('liv.html')

@app.route('/cancer_assessment/bladderentry')
def bladderentry():
    return render_template('bladderentry.html')

@app.route('/cancer_assessment/bileductentry')
def bileductentry():
    return render_template('bileductentry.html')

@app.route('/cancer_assessment/breastentry')
def breastentry():
    return render_template('breastentry.html')

@app.route('/cancer_assessment/lungentry')
def lungentry():
    return render_template('lungentry.html')

@app.route('/cancer_assessment/breastmediillus')
def breastmediillus():
    return render_template('breastmediillus.html')



@app.route('/cancer_assessment/lungmediillus')
def lungmediillus():
    return render_template('lungmediillus.html')

@app.route('/cancer_assessment/bileductmediillus')
def bileductmediillus():
    return render_template('bileductmediillus.html')

@app.route('/cancer_assessment/livmediillus')
def livmediillus():
    return render_template('livmediillus.html')


@app.route('/cancer_assessment/bileducttype')
def bileducttype():
    return render_template('bileducttype.html')

@app.route('/cancer_assessment/bilesym')
def bilesym():
    return render_template('bilesym.html')

@app.route('/cancer_assessment/biletreatments')
def biletreatments():
    return render_template('biletreatments.html')

@app.route('/cancer_assessment/bilestat')
def bilestat():
    return render_template('bilestat.html')

@app.route('/cancer_assessment/bilestages')
def bilestages():
    return render_template('bilestages.html')

@app.route('/cancer_assessment/bileriskprev')
def bileriskprev():
    return render_template('bileriskprev.html')

@app.route('/cancer_assessment/bilediag')
def bilediag():
    return render_template('bilediag.html')


@app.route('/cancer_assessment/livtypes')
def livtypes():
    return render_template('livtypes.html')

@app.route('/cancer_assessment/livsym')
def livsym():
    return render_template('livsym.html')

@app.route('/cancer_assessment/livtreatments')
def livtreatments():
    return render_template('livtreatments.html')

@app.route('/cancer_assessment/livstat')
def livstat():
    return render_template('livstat.html')

@app.route('/cancer_assessmlivstages')
def livstages():
    return render_template('livstages.html')

@app.route('/cancer_assessmlivriskprev')
def livriskprev():
    return render_template('livriskprev.html')

@app.route('/cancer_assessmlivdiag')
def livdiag():
    return render_template('livdiag.html')




@app.route('/cancer_assessment/breasttypes')
def breasttypes():
    return render_template('breasttypes.html')

@app.route('/cancer_assessment/breastsym')
def breastsym():
    return render_template('breastsym.html')

@app.route('/cancer_assessment/breasttreatments')
def breasttreatments():
    return render_template('breasttreatments.html')

@app.route('/cancer_assessment/breaststat')
def breaststat():
    return render_template('breaststat.html')

@app.route('/cancer_assessment/breaststages')
def breaststages():
    return render_template('breaststages.html')

@app.route('/cancer_assessment/breastriskprev')
def breastriskprev():
    return render_template('breastriskprev.html')

@app.route('/cancer_assessment/breastdiag')
def breastdiag():
    return render_template('breastdiag.html')


@app.route('/cancer_assessment/pancmediillus')
def pancmediillus():
    return render_template('pancmediillus.html')


@app.route('/cancer_assessment/panctypes')
def panctypes():
    return render_template('panctypes.html')

@app.route('/cancer_assessment/pancsym')
def pancsym():
    return render_template('pancsym.html')

@app.route('/cancer_assessment/panctreatments')
def panctreatments():
    return render_template('panctreatments.html')

@app.route('/cancer_assessment/pancstat')
def pancstat():
    return render_template('pancstat.html')

@app.route('/cancer_assessment/pancstages')
def pancstages():
    return render_template('pancstages.html')

@app.route('/cancer_assessment/pancriskprev')
def pancriskprev():
    return render_template('pancriskprev.html')

@app.route('/cancer_assessment/pancdiag')
def pancdiag():
    return render_template('pancdiag.html')


@app.route('/cancer_assessment/contactus' , methods=['GET','POST'])
def contactus():
    if request.method == 'POST':
        ph=request.form['ph']
        mail=request.form['mail']
        name=request.form['name']
        type_of_cancer=request.form['type_of_cancer']
        age=request.form['age']
        cursor = mysql.connection.cursor()
        cursor.execute(
            'insert into contactus values(%s,%s,%s,%s,%s)',(ph,mail,name,type_of_cancer,age)
        )
        mysql.connection.commit()
        cursor.close()
    return render_template('contactus.html')

@app.route('/cancer_assessment/terms')
def terms():
    return render_template('terms.html')

@app.route('/cancer_assessment/home')
def home():
    return render_template('home.html')

@app.route('/cancer_assessment/test')
def test():
    return render_template('test.html')

@app.route('/cancer_assessment/kitform' , methods=['GET','POST'])
def kitform():
    if request.method == 'POST':
        ph=request.form['ph']
        mail=request.form['mail']
        name=request.form['name']
        address=request.form['address']
        age=request.form['age']
        cursor = mysql.connection.cursor()
        cursor.execute(
            'insert into kit_orders values(%s,%s,%s,%s,%s)',(ph,mail,name,address,age)
        )
        mysql.connection.commit()
        cursor.close()
    return render_template('kitform.html')

@app.route('/cancer_assessment/bladder' , methods=['GET','POST'])
def bladder():
    """if request.method == 'POST':
        ph=request.form['ph']
        mail=request.form['mail']
        name=request.form['name']
        address=request.form['address']
        age=request.form['age']
        cursor = mysql.connection.cursor()
        cursor.execute(
            'insert into commontypes values(%s,%s,%s,%s,%s)',(ph,mail,name,address,age)
        )
        mysql.connection.commit()
        cursor.close()"""
    return render_template('bladder.html')

@app.route('/cancer_assessment/breast')
def breast():
    
    return render_template('breast.html')

@app.route('/cancer_assessment/lung')
def lung():
    
    return render_template('lung.html')


@app.route('/cancer_assessment/bileduct')
def bileduct():
    return render_template('bileduct.html')

@app.route('/cancer_assessment/compensation')
def compensation():
    return render_template('compensation.html')


@app.route('/cancer_assessment/login', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
        # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('main'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'

    return render_template('index.html', msg='')

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/cancer_assessment/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('withoutlogin'))

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/cancer_assessment/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (%s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/cancer_assessment/home2')
def home2():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home2.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/cancer_assessment/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (session['username'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))



#riskassess

@app.route("/typesofcancer")

def typesofcancer():
    return render_template("typesofcancer.html")

@app.route("/main")

def main():
    return render_template("main.html")

@app.route("/dashboard")

def dashboard():
    return render_template("dashboard.html")



@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

@app.route("/orderplaced")

def orderplaced():
    return render_template("orderplaced.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)
    if(int(result) == 1):
        prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Kidney Disease"
    return render_template("kidney_result.html", prediction_text=prediction)


@app.route("/liver")

def liver():
    return render_template("liver.html")


def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load('liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predictliver', methods=["POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)

    if int(result) == 1:
        prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Liver Disease"
    return render_template("liver_result.html", prediction_text=prediction)





@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"

    return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))


##################################################################################

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('diab_result.html', prediction=my_prediction)


############################################################################################################
















if __name__=="__main__":
    app.run(debug=True)

    """CREATE TABLE `cancer_assessment`.`contactus` (`phone number` TEXT NOT NULL , `mail` TEXT NOT NULL , `name` TEXT NOT NULL , `type_of_cancer` TEXT NOT NULL , `age` INT(2) NOT NULL ) ENGINE = InnoDB;"""