from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load the pre-trained model
model = joblib.load('model.pkl')

# Define the features expected by the model
FEATURES = [
    'Recipientgender', 'Stemcellsource', 'Donorage', 'Donorage35', 'IIIV', 
    'Gendermatch', 'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 
    'CMVstatus', 'DonorCMV', 'RecipientCMV', 'Disease', 'Riskgroup', 
    'Txpostrelapse', 'Diseasegroup', 'HLAmatch', 'HLAmismatch', 'Antigen', 
    'Allele', 'Recipientage10', 'Recipientageint', 'Relapse', 'aGvHDIIIIV', 
    'extcGvHD', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass', 
    'ANCrecovery', 'PLTrecovery', 'time_to_aGvHD_III_IV', 'survival_time'
]

# Descriptions for each feature
FEATURE_DESCRIPTIONS = {
    'Recipientgender': 'Male - 1, Female - 0',
    'Stemcellsource': 'Peripheral blood - 1, Bone marrow - 0',
    'Donorage': 'Age of the donor (numeric)',
    'Donorage35': 'Donor age <35 - 0, Donor age >=35 - 1',
    'IIIV': 'Acute graft vs host disease stage II/III/IV (Yes - 1, No - 0)',
    'Gendermatch': 'Female to Male - 1, Other - 0',
    'DonorABO': '0 (0), 1 (A), -1 (B), 2 (AB)',
    'RecipientABO': '0 (0), 1 (A), -1 (B), 2 (AB)',
    'RecipientRh': '1 (+), 0 (-)',
    'ABOmatch': '1 (Matched), 0 (Mismatched)',
    'CMVstatus': 'Numeric (higher value = lower compatibility)',
    'DonorCMV': '1 (Presence), 0 (Absence)',
    'RecipientCMV': '1 (Presence), 0 (Absence)',
    'Disease': 'Numeric (e.g., 0, 1, 2, etc.)',
    'Riskgroup': '1 (High risk), 0 (Low risk)',
    'Txpostrelapse': '1 (Yes), 0 (No)',
    'Diseasegroup': '1 (Malignant), 0 (Nonmalignant)',
    'HLAmatch': '0 (10/10), 1 (9/10), 2 (8/10), 3 (7/10)',
    'HLAmismatch': '0 (Matched), 1 (Mismatched)',
    'Antigen': '-1 (No differences), 0 (One difference), 1 (Two differences), 2 (Three differences)',
    'Allele': '-1 (No differences), 0 (One difference), 1 (Two differences), 2 (Three differences), 3 (Four differences)',
    'Recipientage10': '0 (<10), 1 (>=10)',
    'Recipientageint': '0 (0-5), 1 (5-10), 2 (10-20)',
    'Relapse': '1 (Yes), 0 (No)',
    'aGvHDIIIIV': '1 (Yes), 0 (No)',
    'extcGvHD': '1 (Yes), 0 (No)',
    'CD34kgx10d6': 'Numeric (e.g., 5.08)',
    'CD3dCD34': 'Numeric (e.g., 0.709805)',
    'CD3dkgx10d8': 'Numeric (e.g., 7.16)',
    'Rbodymass': 'Numeric (e.g., 54.1)',
    'ANCrecovery': 'Numeric (e.g., 15.0)',
    'PLTrecovery': 'Numeric (e.g., 17.0)',
    'time_to_aGvHD_III_IV': 'Numeric (e.g., 1000000.0)',
    'survival_time': "Numeric (e.g., 20.0)"
}

# Fake login credentials (for demonstration only)
VALID_USERNAME = "doctor"
VALID_PASSWORD = "password"

@app.route('/')
def home():
    # Redirect to login page if not logged in
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            # Store username in session
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Remove username from session
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/index', methods=['GET', 'POST'])
def index():
    # Redirect to login page if not logged in
    if 'username' not in session:
        return redirect(url_for('login'))
    
    result = None
    if request.method == 'POST':
        # Get input data from the form
        input_data = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(input_array)[0]
        result = "Live (Success)" if prediction == 0 else "Die (Failure)"
    
    return render_template('index.html', features=FEATURES, feature_descriptions=FEATURE_DESCRIPTIONS, result=result)

if __name__ == '__main__':
    app.run(debug=True)