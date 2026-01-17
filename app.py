from flask import Flask, render_template, request
import os
import pandas as pd
from utils.chatgpt_api import get_full_analysis
from utils.plot_utils import generate_feature_plot, generate_demographic_plot

# --- FLASK SETUP ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- ROUTES ---
@app.route("/")
def index():
    """Renders the project introduction and narrative page."""
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Handles both GET (show form) and POST (process data, get prediction, show results).
    """
    if request.method == "POST":
        file = request.files.get("file")
        df = pd.DataFrame()
        
        # 1. Handle CSV Upload
        if file and file.filename.endswith(".csv"):
            try:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                df = pd.read_csv(filepath)
                
                # FIXED: Handle categorical columns - convert text to numeric where needed
                # Common mappings for HR data
                categorical_mappings = {
                    'Very Low': 1, 'Low': 1,
                    'Medium': 2, 'Moderate': 2,
                    'High': 3,
                    'Very High': 4, 'Excellent': 4,
                    'Yes': 1, 'No': 0,
                    'Male': 1, 'Female': 0,
                    'Poor': 1, 'Fair': 2, 'Good': 3, 'Very Good': 4
                }
                
                # Apply mappings to all columns
                for col in df.columns:
                    if df[col].dtype == 'object':  # If column contains text
                        # Try to map categorical values to numbers
                        df[col] = df[col].map(lambda x: categorical_mappings.get(str(x).strip(), x) if pd.notna(x) else x)
                        
                        # If still contains strings after mapping, keep as-is for LLM to interpret
                        # (like CustomFeature values or department names)
                
            except Exception as e:
                return f"Error reading CSV file: {e}", 500
        
        # 2. Handle Manual Input (NOW WITH CUSTOM FEATURE VALUE)
        elif not file or not file.filename:
            data = {
                "Age": request.form.get("age"),
                "YearsAtCompany": request.form.get("years"),
                "MonthlyIncome": request.form.get("income"),
                "JobSatisfaction": request.form.get("jobsat"),
                "WorkLifeBalance": request.form.get("wlb"),
                "OverTimeHours": request.form.get("ot"),
            }
            
            # FIXED: Handle custom feature properly with both name AND value
            custom_feature_name = request.form.get("custom_feature", "").strip()
            custom_feature_value = request.form.get("custom_feature_value", "").strip()
            
            # Only add custom feature if BOTH name and value are provided
            if custom_feature_name and custom_feature_value:
                data["CustomFeature"] = custom_feature_name
                data["CustomFeatureValue"] = custom_feature_value
            
            # Filter out empty inputs and create DataFrame
            valid_data = {k: [v] for k, v in data.items() if v is not None and v != ''}
            
            if valid_data:
                df = pd.DataFrame(valid_data)
            else:
                return "No input data provided. Please fill out the form.", 400
        
        if df.empty:
            return "Please provide valid data via CSV or form.", 400
        
        # 3. Call the LLM Backend for Analysis
        try:
            full_analysis = get_full_analysis(df)
            
            # 4. Generate Plots with error handling
            feature_plot = None
            demo_plot = None
            
            try:
                if full_analysis.get('feature_importance'):
                    feature_plot = generate_feature_plot(full_analysis['feature_importance'])
            except Exception as e:
                print(f"Error generating feature plot: {e}")
            
            try:
                if full_analysis.get('demographics_plot_data'):
                    demo_plot = generate_demographic_plot(full_analysis['demographics_plot_data'])
            except Exception as e:
                print(f"Error generating demographic plot: {e}")
            
            # 5. Split data into High Risk and Low Risk tables
            predictions_dict = {p['name']: p['prediction'] for p in full_analysis.get('predictions', [])}
            
            # Add a 'Risk' column to the dataframe for filtering
            df['Attrition_Risk'] = df.apply(
                lambda row: predictions_dict.get(row.get('Name', f"Employee_{row.name+1}"), 'No'),
                axis=1
            )
            
            # Split into high and low risk dataframes
            high_risk_df = df[df['Attrition_Risk'] == 'Yes'].copy()
            low_risk_df = df[df['Attrition_Risk'] == 'No'].copy()
            
            # Remove the temporary Risk column before displaying
            high_risk_df = high_risk_df.drop(columns=['Attrition_Risk'])
            low_risk_df = low_risk_df.drop(columns=['Attrition_Risk'])
            
            # Convert to HTML tables
            high_risk_table = high_risk_df.to_html(
                classes="table table-striped table-bordered table-hover table-danger", 
                index=False
            ) if not high_risk_df.empty else ""
            
            low_risk_table = low_risk_df.to_html(
                classes="table table-striped table-bordered table-hover table-success", 
                index=False
            ) if not low_risk_df.empty else ""
            
            # 6. Render Results Page
            return render_template("results.html",
                predictions=full_analysis.get('predictions', []),
                high_risk_count=len(high_risk_df),
                low_risk_count=len(low_risk_df),
                high_risk_table=high_risk_table,
                low_risk_table=low_risk_table,
                feature_plot=feature_plot,
                demo_plot=demo_plot
            )
        
        except Exception as e:
            return f"Error during analysis: {e}", 500
    
    # If GET request, show the input form
    return render_template("predict_form.html")

# --- EXECUTION ---
if __name__ == "__main__":
    app.run(debug=True)