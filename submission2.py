import pandas as pd
import numpy as np
import random
import joblib



# Load pre-trained models and other utilities
scaler = joblib.load('scaler.pkl')  # For ESG predictor
stacking_model = joblib.load('stacking_model.pkl')  # ESG model
increment_diff_model = joblib.load('stock_prediction_model_RFC_3.pkl')  # Incremento-Diff RFC
stock_scaler = joblib.load('stock_scaler.pkl')  # For Incremento-Diff model
best_threshold = 0.82

# Define encoders and feature order
encoders = {
    'industry': joblib.load("industry_encoder.pkl"),
    'environment_grade': joblib.load("environment_grade_encoder.pkl"),
    'environment_level': joblib.load("environment_level_encoder.pkl"),
    'social_grade': joblib.load("social_grade_encoder.pkl"),
    'social_level': joblib.load("social_level_encoder.pkl"),
    'governance_grade': joblib.load("governance_grade_encoder.pkl"),
    'governance_level': joblib.load("governance_level_encoder.pkl"),
    'total_grade': joblib.load("total_grade_encoder.pkl"),
    'total_level': joblib.load("total_level_encoder.pkl"),
}

# Define the exact order of feature names
feature_order = [
    'weighted_score', 'environment_score_norm', 'social_score_norm',
    'governance_score_norm', 'env_social_interaction', 'env_gov_interaction',
    'social_gov_interaction', 'environment_grade', 'environment_level',
    'social_grade', 'social_level', 'governance_grade', 'governance_level',
    'total_grade', 'total_level', 'score_product', 'env_soc_gov_product',
    'total_score_norm', 'score_volatility', 'env_minus_soc', 'env_minus_gov',
    'soc_minus_gov', 'env_soc_level_0_1', 'env_soc_level_0_2',
    'env_soc_level_0_3', 'env_soc_level_1_0', 'env_soc_level_1_1',
    'env_soc_level_1_2', 'env_soc_level_1_3', 'env_soc_level_2_1',
    'env_soc_level_2_3', 'env_gov_level_0_0', 'env_gov_level_0_1',
    'env_gov_level_0_2', 'env_gov_level_1_0', 'env_gov_level_1_1',
    'env_gov_level_1_2', 'env_gov_level_2_0', 'env_gov_level_2_2',
    'industry_0', 'industry_1', 'industry_2', 'industry_3', 'industry_4',
    'industry_5', 'industry_6', 'industry_7', 'industry_8', 'industry_9',
    'industry_10', 'industry_11', 'industry_12', 'industry_13',
    'industry_14', 'industry_15', 'industry_16', 'industry_17',
    'industry_18', 'industry_19', 'industry_20', 'industry_21',
    'industry_22', 'industry_23', 'industry_24', 'industry_25',
    'industry_26', 'industry_27', 'industry_28', 'industry_29',
    'industry_30', 'industry_31', 'industry_32', 'industry_33',
    'industry_34', 'industry_35', 'industry_36', 'industry_37',
    'industry_38', 'industry_39', 'industry_40', 'industry_41',
    'industry_42', 'industry_43', 'industry_44'
]


INDUSTRY_WEIGHTS = {
    # High environmental impact industries
    'Energy': {'environment': 0.5, 'social': 0.25, 'governance': 0.25},
    'Utilities': {'environment': 0.5, 'social': 0.25, 'governance': 0.25},
    'Chemicals': {'environment': 0.45, 'social': 0.3, 'governance': 0.25},
    'Metals and Mining': {'environment': 0.45, 'social': 0.3, 'governance': 0.25},
    # Technology and innovation focused
    'Technology': {'environment': 0.25, 'social': 0.4, 'governance': 0.35},
    'Semiconductors': {'environment': 0.3, 'social': 0.35, 'governance': 0.35},
    'Communications': {'environment': 0.2, 'social': 0.45, 'governance': 0.35},
    'Telecommunication': {'environment': 0.2, 'social': 0.45, 'governance': 0.35},
    # Healthcare and life sciences
    'Biotechnology': {'environment': 0.25, 'social': 0.45, 'governance': 0.3},
    'Health Care': {'environment': 0.25, 'social': 0.45, 'governance': 0.3},
    'Pharmaceuticals': {'environment': 0.3, 'social': 0.4, 'governance': 0.3},
    'Life Sciences Tools and Services': {'environment': 0.25, 'social': 0.45, 'governance': 0.3},
    # Financial sector
    'Financial Services': {'environment': 0.2, 'social': 0.35, 'governance': 0.45},
    'Banking': {'environment': 0.2, 'social': 0.35, 'governance': 0.45},
    'Insurance': {'environment': 0.2, 'social': 0.35, 'governance': 0.45},
    # Consumer-facing industries
    'Retail': {'environment': 0.3, 'social': 0.4, 'governance': 0.3},
    'Consumer products': {'environment': 0.35, 'social': 0.4, 'governance': 0.25},
    'Food Products': {'environment': 0.4, 'social': 0.35, 'governance': 0.25},
    'Beverages': {'environment': 0.4, 'social': 0.35, 'governance': 0.25},
    # Manufacturing and industrial
    'Electrical Equipment': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    'Machinery': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    'Aerospace and Defense': {'environment': 0.35, 'social': 0.3, 'governance': 0.35},
    'Auto Components': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    'Automobiles': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    # Service-based industries
    'Media': {'environment': 0.2, 'social': 0.45, 'governance': 0.35},
    'Professional Services': {'environment': 0.2, 'social': 0.45, 'governance': 0.35},
    'Commercial Services and Supplies': {'environment': 0.25, 'social': 0.4, 'governance': 0.35},
    # Real estate and construction
    'Real Estate': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    'Building': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    'Construction': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    # Transportation
    'Airlines': {'environment': 0.45, 'social': 0.3, 'governance': 0.25},
    'Marine': {'environment': 0.45, 'social': 0.3, 'governance': 0.25},
    'Road and Rail': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    'Logistics and Transportation': {'environment': 0.4, 'social': 0.3, 'governance': 0.3},
    # Default weights for other industries
    'Blank Check Companies': {'environment': 0.33, 'social': 0.33, 'governance': 0.34}
}

def get_industry_weights(industry):
    return INDUSTRY_WEIGHTS.get(industry, INDUSTRY_WEIGHTS['Blank Check Companies'])

# Define helper functions
def calculate_weighted_score(row):
    weights = get_industry_weights(row['industry'])
    env_score = row['environment_score'] * weights['environment']
    social_score = row['social_score'] * weights['social']
    gov_score = row['governance_score'] * weights['governance']
    # Adjust weights based on grades
    env_score *= (1 + row['environment_grade'] * 0.1)
    social_score *= (1 + row['social_grade'] * 0.1)
    gov_score *= (1 + row['governance_grade'] * 0.1)
    return env_score + social_score + gov_score

def preprocess_input_2(input_features: dict, encoders: dict) -> pd.DataFrame:
    """
    Preprocess input features to apply encoders and calculate derived features automatically.

    Args:
    - input_features (dict): Input features provided by the user.
    - encoders (dict): Encoders for categorical features like grades and levels.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame with encoded and derived features.
    """
    # Convert the input features to a DataFrame
    df = pd.DataFrame([input_features])
    
    # Lowercase all string features except 'industry'
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'industry':
            df[col] = df[col].str.lower()
    
    # Apply encoders to categorical fields
    for feature, encoder in encoders.items():
        if feature in df:
            df[feature] = encoder.transform(df[[feature]].astype(str))  # Reshape as 2D array or DataFrame
    
    # Handle missing or blank industries
    df['industry'] = df['industry'].fillna('Blank Check Companies')
    
    # Calculate weighted score
    df['weighted_score'] = df.apply(calculate_weighted_score, axis=1)
    
    # Normalize numeric scores using StandardScaler
    # numeric_cols = ['environment_score', 'social_score', 'governance_score']
    df['environment_score_norm'] = joblib.load('environment_score_scaler.pkl').transform(df[['environment_score']])
    df['social_score_norm'] = joblib.load('social_score_scaler.pkl').transform(df[['social_score']])
    df['governance_score_norm'] = joblib.load('governance_score_scaler.pkl').transform(df[['governance_score']])
    
    # Create interaction features
    df['env_social_interaction'] = df['environment_score_norm'] * df['social_score_norm']
    df['env_gov_interaction'] = df['environment_score_norm'] * df['governance_score_norm']
    df['social_gov_interaction'] = df['social_score_norm'] * df['governance_score_norm']
    
    return df



def feature_engineering_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced features and one-hot encode categorical columns.
    """
    # Interaction-based features
    df['score_product'] = df['weighted_score'] * df['total_level']
    df['env_soc_gov_product'] = (
        df['environment_score_norm'] 
        * df['social_score_norm'] 
        * df['governance_score_norm']
    )
    
    # Averaged score
    df['total_score_norm'] = (
        df['environment_score_norm'] 
        + df['social_score_norm'] 
        + df['governance_score_norm']
    ) / 3.0
    
    # Volatility
    df['score_volatility'] = np.std(
        [
            df['environment_score_norm'], 
            df['social_score_norm'], 
            df['governance_score_norm']
        ],
        axis=0
    )
    
    # Differences
    df['env_minus_soc'] = df['environment_score_norm'] - df['social_score_norm']
    df['env_minus_gov'] = df['environment_score_norm'] - df['governance_score_norm']
    df['soc_minus_gov'] = df['social_score_norm'] - df['governance_score_norm']
    
    # Combine categorical levels
    df['env_soc_level'] = df['environment_level'].astype(str) + '_' + df['social_level'].astype(str)
    df['env_gov_level'] = df['environment_level'].astype(str) + '_' + df['governance_level'].astype(str)
    
    # One-hot encode categorical features
    categorical_cols = ['env_soc_level', 'env_gov_level', 'industry']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    return df_encoded


# def predict_pipeline_2(input_features: dict, encoders: dict) -> int:
#     """
#     Predict the target class using the trained stacking model and input features.

#     Args:
#     - input_features (dict): A dictionary containing input features.
#     - encoders (dict): Encoders for categorical features like grades and levels.

#     Returns:
#     - int: Predicted class (0 or 1).
#     """
#     # Preprocess the input to apply encoders and calculate derived features
#     input_df = preprocess_input_2(input_features, encoders)
    
#     # Perform feature engineering
#     input_df_engineered = feature_engineering_2(input_df)
    
#     # Align the engineered features with the specified feature order
#     for col in feature_order:
#         if col not in input_df_engineered:
#             input_df_engineered[col] = 0  # Add missing columns with default value
#     input_df_engineered = input_df_engineered[feature_order]  # Reorder columns
    
#     # Scale the features
#     input_scaled = scaler.transform(input_df_engineered)
    
#     # Get probabilities
#     prob_pos = stacking_model.predict_proba(input_scaled)[:, 1][0]
    
#     # Apply the best threshold
#     predicted_class = int(prob_pos >= best_threshold)
    
#     return predicted_class

def predict_pipeline_2(input_features: dict, encoders: dict):
    """
    Predict the target class using the trained stacking model and input features.

    Args:
        input_features (dict): A dictionary containing input features.
        encoders (dict): Encoders for categorical features like grades and levels.

    Returns:
        (float, int): Tuple containing probability of the positive class and predicted class (0 or 1).
    """
    # Preprocess the input
    input_df = preprocess_input_2(input_features, encoders)

    # Feature engineering
    input_df_engineered = feature_engineering_2(input_df)

    # Align with feature order
    for col in feature_order:
        if col not in input_df_engineered:
            input_df_engineered[col] = 0
    input_df_engineered = input_df_engineered[feature_order]

    # Scale
    input_scaled = scaler.transform(input_df_engineered)

    # Get probability
    prob_pos = stacking_model.predict_proba(input_scaled)[:, 1][0]

    # Get predicted class
    predicted_class = int(prob_pos >= best_threshold)

    # Return both the probability and the prediction
    return prob_pos, predicted_class


def predict_increment_diff(today_price, yesterday_price):
    """Make predictions using the Incremento-Diff model."""
    features = pd.DataFrame({
        'INCREMENTO': [((today_price - yesterday_price)/yesterday_price)],
        'diff': [today_price - yesterday_price]
    })
    features_scaled = stock_scaler.transform(features)
    # print(features_scaled)
    prob_pos = increment_diff_model.predict_proba(features_scaled)[:, 1][0]
    return prob_pos, int(prob_pos >= 0.5)

# def predict_increment_diff(today_price, yesterday_price):
#     features = pd.DataFrame({
#         'INCREMENTO': [(today_price - yesterday_price)/yesterday_price],
#         'diff': [today_price - yesterday_price]
#     })
#     features_scaled = stock_scaler.transform(features)
#     prob_pos = increment_diff_model.predict_proba(features_scaled)[:, 1][0]
#     pred = int(prob_pos >= 0.5)
#     return prob_pos, pred


def input_features_from_user() -> dict:
    """
    Collect input features from the user one by one.

    Returns:
    - dict: A dictionary of input features.
    """
    features = {}
    features['industry'] = input("Enter the industry: ")
    features['environment_grade'] = input("Enter the environment grade (e.g., A, B, etc.): ")
    features['environment_level'] = input("Enter the environment level (e.g., High, Medium, etc.): ")
    features['social_grade'] = input("Enter the social grade (e.g., A, B, etc.): ")
    features['social_level'] = input("Enter the social level (e.g., High, Medium, etc.): ")
    features['governance_grade'] = input("Enter the governance grade (e.g., A, B, etc.): ")
    features['governance_level'] = input("Enter the governance level (e.g., High, Medium, etc.): ")
    features['total_grade'] = input("Enter the total grade (e.g., A, B, etc.): ")
    features['total_level'] = input("Enter the total level (e.g., High, Medium, etc.): ")
    features['environment_score'] = float(input("Enter the environment score: "))
    features['social_score'] = float(input("Enter the social score: "))
    features['governance_score'] = float(input("Enter the governance score: "))
    features['total_score'] = float(input("Enter the total score: "))
    return features


def guardrail_combined_predict(input_features, today_price, yesterday_price):
    """
    Combine ESG and Incremento-Diff models using guardrail rules.

    Args:
    - input_features (dict): User input features for ESG model.
    - today_price (float): Today's stock price.
    - yesterday_price (float): Yesterday's stock price.

    Returns:
    - str: Final decision ('Buy' or 'Don't Buy').
    """
    # Get predictions from ESG model
    esg_prob, esg_pred = predict_pipeline_2(input_features, encoders)

    # Get predictions from Incremento-Diff model
    inc_prob, inc_pred = predict_increment_diff(today_price, yesterday_price)

    # Guardrail rules
    # Rule 1: Strong ESG signal overrides everything
    if esg_prob > 0.9:
        return 'Buy'

    # Rule 2: Volatility check to prevent buying during extreme spikes
    if today_price > 1.5 * yesterday_price:
        return 'Don\'t Buy'

    # Rule 3: If both models agree, trust the prediction
    if esg_pred == 1 and inc_pred == 1:
        return 'Buy'
    if esg_pred == 0 and inc_pred == 0:
        return 'Don\'t Buy'

    # Rule 4: Confidence in ESG model takes precedence
    if esg_prob >= 0.8:
        return 'Buy' if esg_pred == 1 else 'Don\'t Buy'

    # Rule 5: Incremento-Diff acts as a fallback when ESG confidence is low
    return 'Buy' if inc_pred == 1 else 'Don\'t Buy'


def main():
    print("Welcome to the Combined Prediction Terminal!")
    
    try:
        print("\nPlease provide ESG-related input features:")
        user_features = input_features_from_user()  # Function to collect ESG-related input features
        today_price = float(input("Enter today's stock price: "))  # Collect today's stock price
        yesterday_price = float(input("Enter yesterday's stock price: "))  # Collect yesterday's stock price
        
        # Make the combined prediction using the guardrail mechanism
        decision = guardrail_combined_predict(user_features, today_price, yesterday_price)
        
        print(f"\nFinal Decision: {decision}")
        
        
        
        # today_price = 1000.877567
        # yesterday_price = 100.347324

        # inc_prob, inc_pred = predict_increment_diff(today_price, yesterday_price)
        # print(f"Incremento-Diff Probability: {inc_prob}")
        # print(f"Incremento-Diff Prediction: {'Buy' if inc_pred == 1 else 'Dont Buy'}")
        
        
        # input_features = {
        #     "industry": "Hotels Restaurants and Leisure",
        #     "environment_grade": "A",
        #     "environment_level": "High",
        #     "social_grade": "BB",
        #     "social_level": "Medium",
        #     "governance_grade": "BB",
        #     "governance_level": "Medium",
        #     "environment_score": 500,
        #     "social_score": 320,
        #     "governance_score": 310,
        #     "total_score": 1130,
        #     "total_grade": "BBB",
        #     "total_level": "High"
        # }  # Replace with sample inputs
        # today_price = 110
        # yesterday_price = 100

        # decision = guardrail_combined_predict(input_features, today_price, yesterday_price)
        # print("Final Decision:", decision)


    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
























# def guardrail_combined_predict(input_features, today_price, yesterday_price):
#     """
#     Combine ESG and Incremento-Diff models using guardrail rules.

#     Args:
#     - input_features (dict): User input features for ESG model.
#     - today_price (float): Today's stock price.
#     - yesterday_price (float): Yesterday's stock price.

#     Returns:
#     - str: Final decision ('Buy' or 'Don't Buy').
#     """
#     esg_prob, esg_pred = predict_esg(input_features)
#     inc_prob, inc_pred = predict_increment_diff(today_price, yesterday_price)

#     # Guardrail rules
#     if esg_prob > 0.9:  # Rule 1: Strong ESG signal overrides everything
#         return 'Buy'
#     if inc_pred == 1 and today_price > 1.5 * yesterday_price:  # Rule 2: Volatility check
#         return 'Don\'t Buy'
#     if esg_pred == 1 and inc_pred == 1:  # Rule 3: If both agree, recommend Buy
#         return 'Buy'
#     if esg_pred == 0 and inc_pred == 0:  # Rule 4: If both disagree, recommend Don't Buy
#         return 'Don\'t Buy'
#     if esg_prob < 0.6:  # Rule 5: If ESG confidence is low, rely on Incremento-Diff
#         return 'Buy' if inc_pred == 1 else 'Don\'t Buy'

#     # Fallback rule: Prioritize ESG model
#     return 'Buy' if esg_pred == 1 else 'Don\'t Buy'

# # Main function for user interaction
# def main():
#     print("Welcome to the Combined Prediction Terminal!")

#     while True:
#         print("\nPlease choose an option:")
#         print("1. Use random samples from dataset")
#         print("2. Provide features")
#         print("3. Exit")

#         choice = input("Enter your choice (1/2/3): ")

#         if choice == "1":
#             try:
#                 csv_file_path = "Final_Data.csv"  # Path to CSV
#                 data = pd.read_csv(csv_file_path)

#                 random_rows = data.sample(n=20, random_state=random.randint(0, 100000))
#                 for _, row in random_rows.iterrows():
#                     esg_input = row.drop(['today_price', 'yesterday_price']).to_dict()
#                     decision = guardrail_combined_predict(
#                         esg_input, row['today_price'], row['yesterday_price']
#                     )
#                     print(f"Prediction: {decision}")
#             except Exception as e:
#                 print(f"Error: {e}")

#         elif choice == "2":
#             try:
#                 print("\nPlease provide ESG-related input features:")
#                 user_features = input_features_from_user()
#                 today_price = float(input("Enter today's stock price: "))
#                 yesterday_price = float(input("Enter yesterday's stock price: "))
#                 decision = guardrail_combined_predict(user_features, today_price, yesterday_price)
#                 print(f"\nFinal Decision: {decision}")
#             except Exception as e:
#                 print(f"Error: {e}")

#         elif choice == "3":
#             print("Exiting. Goodbye!")
#             break

#         else:
#             print("Invalid choice. Please try again.")

# if __name__ == "__main__":
#     main()
