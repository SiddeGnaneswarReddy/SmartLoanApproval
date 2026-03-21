from flask import Flask, request, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
import pandas as pd

app = Flask(__name__)

# --- DATABASE CONFIGURATION ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loans.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the Database Table
class AssessmentHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    income = db.Column(db.Float)
    loan_amount = db.Column(db.Float)
    credit_history = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    result = db.Column(db.String(20))

# Create the database file if it doesn't exist
with app.app_context():
    db.create_all()

# --- LOAD AI MODELS ---
model = joblib.load('loan_rf_model.pkl')
encoders = joblib.load('loan_encoders.pkl')

# --- WEB ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    # Fetch all past assessments from the database, ordered by newest first
    records = AssessmentHistory.query.order_by(AssessmentHistory.timestamp.desc()).all()
    return render_template('history.html', records=records)

@app.route('/architecture')
def architecture():
    return render_template('architecture.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect('/')

    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])
        
        # Format math
        num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col])
            
        cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        for col in cat_cols:
            df[col] = encoders[col].transform(df[col])
            
        feature_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                         'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        df = df[feature_order]
        
        # Get AI Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        confidence = round(max(probability) * 100, 1)
        
        if prediction == 1:
            result_text = "APPROVED"
            color = "#10b981"
            status_msg = "Applicant meets risk threshold."
        else:
            result_text = "DENIED"
            color = "#ef4444"
            status_msg = "High risk factors detected."

        # --- SAVE TO DATABASE ---
        new_record = AssessmentHistory(
            income=float(data['ApplicantIncome']) + float(data['CoapplicantIncome']),
            loan_amount=float(data['LoanAmount']),
            credit_history="Good" if data['Credit_History'] == "1.0" else "Bad",
            confidence=confidence,
            result=result_text
        )
        db.session.add(new_record)
        db.session.commit()
        # ------------------------
            
        return render_template('index.html', prediction_text=result_text, color=color, confidence=confidence, status_msg=status_msg, pred_val=int(prediction))

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)