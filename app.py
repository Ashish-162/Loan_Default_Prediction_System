from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import traceback
import os

app = Flask(__name__, template_folder='.', static_folder='.')

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed', 'message': str(e)}), 405

# Global variables for model and data
model = None
data = None
model_loaded = False
data_loaded = False

print("\n" + "="*60)
print("INITIALIZING LOAN DEFAULT PREDICTION SYSTEM")
print("="*60)

# Check if files exist
print("\n[INFO] Checking files in current directory:")
print(f"   Current dir: {os.getcwd()}")
print(f"   - loan_data.csv exists: {os.path.exists('loan_data.csv')}")
print(f"   - advanced_loan_default_model.pkl exists: {os.path.exists('advanced_loan_default_model.pkl')}")

def load_model_and_data():
    global model, data, model_loaded, data_loaded
    
    print("\n[INFO] Loading model and data...")
    
    # Load data first
    try:
        if os.path.exists('loan_data.csv'):
            data = pd.read_csv('loan_data.csv')
            data_loaded = True
            print("   [INFO] Dataset loaded successfully")
            print(f"     - Shape: {data.shape}")
            print(f"     - Columns: {list(data.columns)}")
        else:
            print("   [ERROR] loan_data.csv NOT found")
            data_loaded = False
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        data_loaded = False
    
    # Load model
    try:
        if os.path.exists('advanced_loan_default_model.pkl'):
            model = joblib.load('advanced_loan_default_model.pkl')
            model_loaded = True
            print("   [INFO] Model loaded successfully")
        else:
            print("   [ERROR] advanced_loan_default_model.pkl NOT found")
            print("     Run: python advanced_loan_default.py")
            model_loaded = False
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        model_loaded = False
    
    return model_loaded and data_loaded

# Load on startup
load_model_and_data()

print("\n[INFO] Flask app initialized")
print("=" * 60 + "\n")

@app.route('/')
def index():
    """Serve the dashboard"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        print(f"Error serving dashboard: {e}")
        return f"Error loading dashboard: {e}", 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'data_loaded': data_loaded,
        'message': 'System is ready' if model_loaded and data_loaded else 'Waiting for model/data'
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Get system status"""
    status_info = {
        'model_loaded': model_loaded,
        'data_loaded': data_loaded,
        'model_file_exists': os.path.exists('advanced_loan_default_model.pkl'),
        'data_file_exists': os.path.exists('loan_data.csv'),
        'current_directory': os.getcwd()
    }
    
    if data_loaded:
        status_info['dataset_shape'] = str(data.shape)
        status_info['columns'] = list(data.columns)
    
    return jsonify(status_info), 200

@app.route('/predict', methods=['GET','POST','OPTIONS'])
def predict():
    """API endpoint for making predictions"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 204

    if request.method == 'GET':
        return jsonify({'error': 'Use POST with JSON body to get predictions'}), 400

    print("\n" + "="*60)
    print("PREDICTION REQUEST")
    print("="*60)
    
    try:
        # Check if model and data are loaded
        if not model_loaded:
            error_response = {
                'error': 'Model not loaded',
                'details': 'The trained model could not be loaded. Please run: python advanced_loan_default.py',
                'model_file_exists': os.path.exists('advanced_loan_default_model.pkl')
            }
            print(f"[ERROR] Model Error: {error_response}")
            return jsonify(error_response), 500
        
        if not data_loaded:
            error_response = {
                'error': 'Dataset not loaded',
                'details': 'The dataset could not be loaded. Please ensure loan_data.csv exists.',
                'data_file_exists': os.path.exists('loan_data.csv')
            }
            print(f"[ERROR] Data Error: {error_response}")
            return jsonify(error_response), 500
        
        if model is None or data is None:
            error_response = {'error': 'Model or data is None'}
            print(f"[ERROR] None Error: {error_response}")
            return jsonify(error_response), 500
        
        # Get JSON data from request
        input_data = request.json
        
        if input_data is None:
            error_response = {'error': 'No JSON data received'}
            print(f"[ERROR] Input Error: {error_response}")
            return jsonify(error_response), 400
        
        print(f"[INFO] Received data: {input_data}")
        
        # Validate required features
        required_features = ['Income', 'LoanAmount', 'Age', 'CreditScore', 'EmploymentYears']
        missing_features = [f for f in required_features if f not in input_data]
        
        if missing_features:
            error_response = {'error': f'Missing fields: {", ".join(missing_features)}'}
            print(f"[ERROR] Missing Fields: {error_response}")
            return jsonify(error_response), 400
        
        # Create input DataFrame
        try:
            user_input = pd.DataFrame([{
                'Income': float(input_data['Income']),
                'LoanAmount': float(input_data['LoanAmount']),
                'Age': float(input_data['Age']),
                'CreditScore': float(input_data['CreditScore']),
                'EmploymentYears': float(input_data['EmploymentYears'])
            }])
            # Derive debt-to-income ratio exactly as training code
            user_input['Debt_Income_Ratio'] = user_input['LoanAmount'] / (user_input['Income'] + 1)
            print(f"[INFO] Input DataFrame:\n{user_input}")
        except (ValueError, TypeError) as e:
            error_response = {'error': f'Invalid values: {str(e)}'}
            print(f"[ERROR] Type Error: {error_response}")
            return jsonify(error_response), 400
        
        # Make prediction
        try:
            print("[INFO] Making prediction...")
            prediction = model.predict(user_input)[0]
            prediction_proba = model.predict_proba(user_input)[0]
            
            print(f"   Prediction: {prediction}")
            print(f"   Probabilities: {prediction_proba}")
        except Exception as pred_error:
            error_response = {'error': f'Prediction failed: {str(pred_error)}'}
            print(f"[ERROR] Prediction Error: {error_response}")
            print(traceback.format_exc())
            return jsonify(error_response), 500
        
        # Build response
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Default Risk' if prediction == 1 else 'No Default Risk',
            'confidence': float(max(prediction_proba) * 100),
            'default_probability': float(prediction_proba[1] * 100),
            'no_default_probability': float(prediction_proba[0] * 100)
        }
        
        print(f"[INFO] Success: {result['prediction_label']}")
        print("=" * 60 + "\n")
        return jsonify(result), 200
    
    except Exception as e:
        error_response = {'error': f'Server error: {str(e)}'}
        print(f"[ERROR] Unexpected Error: {error_response}")
        print(traceback.format_exc())
        print("=" * 60 + "\n")
        return jsonify(error_response), 500

@app.route('/get-feature-names', methods=['GET'])
def get_feature_names():
    """Return list of features"""
    try:
        if not data_loaded or data is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        features = {
            'all': list(data.columns),
            'prediction_features': [col for col in data.columns if col != 'Loan_Status']
        }
        return jsonify(features), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-data', methods=['GET'])
def get_data():
    """Fetch dataset records with pagination and filtering"""
    try:
        if not data_loaded or data is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        sort_by = request.args.get('sort_by', 'index', type=str)
        sort_order = request.args.get('sort_order', 'asc', type=str)
        
        # Validate pagination
        page = max(1, page)
        per_page = min(100, max(5, per_page))  # Limit per_page to 5-100
        
        # Create a copy with index
        df = data.copy().reset_index(drop=True)
        df.index.name = 'Record_ID'
        df = df.reset_index()
        
        # Sorting
        if sort_by in df.columns and sort_by != 'Record_ID':
            ascending = sort_order.lower() == 'asc'
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Pagination
        total_records = len(df)
        total_pages = (total_records + per_page - 1) // per_page
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        paginated_df = df.iloc[start_idx:end_idx]
        
        # Convert to list of dictionaries
        records = paginated_df.to_dict('records')
        
        # Convert numeric values to ensure JSON serialization
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value) if isinstance(value, np.floating) else int(value)
        
        return jsonify({
            'success': True,
            'page': page,
            'per_page': per_page,
            'total_records': total_records,
            'total_pages': total_pages,
            'records': records,
            'columns': list(df.columns)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-data-stats', methods=['GET'])
def get_data_stats():
    """Get statistics about the dataset"""
    try:
        if not data_loaded or data is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        df = data.copy()
        
        stats = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': {}
        }
        
        # Calculate numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats['numeric_stats'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
        
        # Get class distribution if Loan_Status exists
        if 'Loan_Status' in df.columns:
            stats['class_distribution'] = df['Loan_Status'].value_counts().to_dict()
        
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-data-record/<int:record_id>', methods=['GET'])
def get_data_record(record_id):
    """Get a specific data record"""
    try:
        if not data_loaded or data is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        if record_id < 0 or record_id >= len(data):
            return jsonify({'error': f'Record ID {record_id} not found'}), 404
        
        record = data.iloc[record_id].to_dict()
        
        # Convert numpy types to Python types
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                record[key] = float(value) if isinstance(value, np.floating) else int(value)
        
        return jsonify({
            'success': True,
            'record_id': record_id,
            'record': record
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if model_loaded and data_loaded:
        print("✓ Starting Flask server on port 5000...")
        print("  Open http://localhost:5000 in your browser")
    else:
        print("⚠️  WARNING: Model and/or data not loaded!")
        print("   The app will run but predictions will fail.")
    
    app.run(debug=True, port=5000)
