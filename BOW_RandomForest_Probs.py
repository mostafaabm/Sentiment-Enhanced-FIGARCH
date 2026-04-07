
"""
@author: mostafaabm
"""

from __future__ import print_function
import gc
gc.collect()
import pandas as pd
import numpy as np
import os
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import timedelta, date, datetime
import yfinance as yf
import math
import requests
import csv
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import re
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas import ExcelWriter
from sklearn.ensemble import RandomForestClassifier


def Window_separation(df): 
    
    df = df.dropna(subset=['time_published']) 
    
    # Step 1: Parse time_published into a full datetime column
    df['datetime_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
    
    # Step 2: Extract date and time separately
    df['date'] = df['datetime_published'].dt.date
    df['time'] = df['datetime_published'].dt.time
    
    # Define market open and close time (example: 9:30 AM - 4:00 PM)
    market_open_hour = 9
    market_open_minute = 30
    
    market_close_hour = 16
    market_close_minute = 00
    
    
    
    def get_window_Open_datetime(row):
        """Calculate start of window for each headline (previous day's market open)"""
        dt = row['datetime_published']
        market_open_today = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                                         hour=market_open_hour, minute=market_open_minute)
    
        if dt >= market_open_today:
            return market_open_today
        else:
            # If before market open today, window started at market open yesterday
            prev_day = market_open_today - pd.Timedelta(days=1)
            return prev_day
    
    
    def get_window_Close_datetime(row):
        """Calculate end of window for each headline (next day's market close)"""
        dt = row['datetime_published']
        market_close_today = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                                         hour=market_close_hour, minute=market_close_minute)
    
        if dt <= market_close_today:
            return market_close_today
        else:
            # If after market close today, window started at market close today
            next_day = market_close_today + pd.Timedelta(days=1)
            return next_day
    
    
    
    # Step 3: Calculate window start and end for each headline
    df['window_start'] = df.apply(get_window_Open_datetime, axis=1)
    df['window_end'] = df.apply(get_window_Close_datetime, axis=1)
    
    # Step 4: Group headlines by window_start and window_end date and predict next day's return
    # Example: group headlines published between each window_start and window_start+1d
    
    df['window_Open_to_Open'] = df['window_start'].dt.date
    df['window_Close_to_Close'] = df['window_end'].dt.date
    
    return df




def generate_bow(ticker):
    dff = pd.read_csv(f"{ticker}_Alpha_FinBERT_dataset.csv")
    df = Window_separation(dff)

    
    ###############################################################
    # Importing frequency-based dataset to merge with BOW
    df2 = pd.read_csv(f"/Users/mostafamoghadam/Documents/Documents - Mostafa’s MacBook Pro/PhD-Wilfrid Laurier University/Pr. Makarov papers (Research)/Thesis/1/Web Scraping/AlphaVantage/Alpha_FinBERT_datasets/Frequency-based_Datasets/{ticker}_freq_dataset.csv")
    df2['date'] = pd.to_datetime(df2['date'])
    
    #################Get market data########################
    MIN_DATE = min(df.date)
    TODAY=max(df.date)
    
    text = df['title'].tolist()
    
    
    # Load spaCy English model
    nlp = spacy.load('en_core_web_sm')
    
    # Function to map NLTK POS tags to WordNet POS tags
    def get_wordnet_pos_spacy(spacy_pos):
        """Map spaCy POS tag to WordNet POS tag."""
        if spacy_pos.startswith('J'):
            return wordnet.ADJ
        elif spacy_pos.startswith('N'):
            return wordnet.NOUN
        elif spacy_pos.startswith('V'):
            return wordnet.VERB
        elif spacy_pos.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    # Custom stopwords
    custom_stopwords = set(stopwords.words('english'))
    additional_stopwords = set([".com","cryptos","crypto","cryptocurrency","$","in", "for", "etc.", "'s", "one", " ", 'e', 'g', 'u', 'wo', 'take', 'get', 'make', 'back', 'like', 'another'])  # Add more if needed
    stopwords_list = list(custom_stopwords.union(additional_stopwords))
    
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    
    synonym_dict = {
        'american': 'usa',
        'americans': 'usa',
        'us': 'usa',
        'america': 'usa',
        'analysts': 'analyst',
        'sp': 'sp500',
        's&p': 'sp500',
        'alphabet': 'google', 
        'employee':'worker',
        'china': 'chinese', 
        'watch': 'look', 
        '3': 'three', 
        'third': 'three',
        'firm': 'company', 
        'covid19': 'coronavirus', 
        'covid-19': 'coronavirus',
        'pandemic': 'coronavirus',
        'iphones': 'Apple products',
        'iphone': 'Apple products', 
        'ipad': 'Apple products', 
        'tech': 'technology', 
        'technological': 'technology', 
        'wall street': 'wall_street',  # Combine 'wall street' into 'wall_street'
        'donald': 'donald trump', 
        'trump': 'donald trump', 
        'apple_inc.': 'apple',  # Update here to merge 'apple inc.' with 'apple'
        'warren': 'BerkshireHathaway',
        'buffett': 'BerkshireHathaway', 
        'berkshire': 'BerkshireHathaway', 
        'hathaway': 'BerkshireHathaway',
        'benefit': 'profit', 
        'tim_cook': 'apple president',  # Adjust as needed
        'federal': 'government', 
        'bitcoin': 'crypto',
        'may': 'could', 
        'ceo': 'tim_cook', 
        'dow': 'DowJonesIndex', 
        'Jones': 'DowJonesIndex', 
        'grow': 'growth', 
        'nt': 'not', 
        "n't": 'not'
    }
    
    # Function to tokenize and process text
    def custom_tokenizer(doc):
        spacy_doc = nlp(doc)
        tokens = []
    
        # Add named entities with underscores as single tokens
        for ent in spacy_doc.ents:
            ent_token = ent.text.replace(' ', '_').lower()
            if ent_token not in stopwords_list:
                tokens.append(ent_token)
    
        # Add other tokens (excluding punctuation, entities, stopwords)
        for token in spacy_doc:
            token_text = token.text.lower()
            if (token.ent_type_ == ''  # Not part of entity
                and not token.is_punct
                and token_text not in stopwords_list):
                wn_pos = get_wordnet_pos_spacy(token.pos_)
                lemma = lemmatizer.lemmatize(token_text, wn_pos)
                replaced = synonym_dict.get(lemma, lemma)
                if replaced not in stopwords_list:
                    tokens.append(replaced)
    
        return tokens
    
    
    # Initialize CountVectorizer with custom tokenizer and ngram_range=(1, 1)
    ##vectorizer = CountVectorizer(max_features=370, tokenizer=custom_tokenizer, ngram_range=(1, 1))
    vectorizer = TfidfVectorizer(max_features=300, tokenizer=custom_tokenizer)
    
    # Fit and transform the text data
    bow_matrix = vectorizer.fit_transform(text)
    
    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the word frequencies
    word_frequencies = bow_matrix.sum(axis=0)
    
    # Convert word frequencies to a list
    word_frequencies_list = word_frequencies.tolist()[0]
    
    
    
    # Create a dictionary to store word-frequency pairs
    word_frequency_dict = dict(zip(feature_names, word_frequencies_list))
    
    sorted_word_frequency = sorted(word_frequency_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Display the list of words to be included in the bag of words and their corresponding frequencies
    print("List of words to be included in the bag of words and their frequencies (sorted):")
    for word, frequency in sorted_word_frequency:
        print(f"{word}: {frequency}")
    
    # Create a DataFrame of the Bag of Words matrix
    bow_df_OO = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)
    bow_df_OO = bow_df_OO.drop(columns=[col for col in bow_df_OO.columns if col in list(df2.columns)])
    bow_df_CC = bow_df_OO.copy()
    
    
    #bow_df['date'] = df.date.copy()    #This line is replaced by the if statement below
    
    bow_df_OO['date'] = pd.to_datetime(df['window_Open_to_Open'].copy())
    bow_OO = bow_df_OO.columns
 
    bow_df_CC['date'] = pd.to_datetime(df['window_Close_to_Close'].copy())
    bow_CC = bow_df_CC.columns
    
    
    
    
    # Group by date and sum the word counts 
    daily_bow_df_OO = bow_df_OO.groupby('date').sum().reset_index()
    daily_bow_df_CC = bow_df_CC.groupby('date').sum().reset_index()
    
    # Merging datasets
    merged_df_OO = pd.merge(daily_bow_df_OO, df2, on='date', how='inner', suffixes=('', '_df2'))
    merged_df_CC = pd.merge(daily_bow_df_CC, df2, on='date', how='inner', suffixes=('', '_df2'))
    
    # Creating new binary response variables
    
    if ticker == 'commodity':
        merged_df_OO['IP_Pos_OC'] = (merged_df_OO['logreturn'] > merged_df_OO['logreturn'].quantile(0.75)).astype(int)
        merged_df_OO['IP_Neg_OC'] = (merged_df_OO['logreturn'] < merged_df_OO['logreturn'].quantile(0.25)).astype(int)
        merged_df_CC['IP_Pos_OC'] = (merged_df_CC['logreturn'] > merged_df_CC['logreturn'].quantile(0.75)).astype(int)
        merged_df_CC['IP_Neg_OC'] = (merged_df_CC['logreturn'] < merged_df_CC['logreturn'].quantile(0.25)).astype(int)  
    else:
        
        merged_df_OO['IP_Pos_OC'] = (merged_df_OO['logreturn_OC'] > merged_df_OO['logreturn_OC'].quantile(0.75)).astype(int)
        merged_df_OO['IP_Neg_OC'] = (merged_df_OO['logreturn_OC'] < merged_df_OO['logreturn_OC'].quantile(0.25)).astype(int)  
        merged_df_OO['IP_Pos_CC'] = (merged_df_OO['logreturn_CC'] > merged_df_OO['logreturn_CC'].quantile(0.75)).astype(int)
        merged_df_OO['IP_Neg_CC'] = (merged_df_OO['logreturn_CC'] < merged_df_OO['logreturn_CC'].quantile(0.25)).astype(int)  
        merged_df_OO['IP_Pos_OO'] = (merged_df_OO['logreturn_OO'] > merged_df_OO['logreturn_OO'].quantile(0.75)).astype(int)
        merged_df_OO['IP_Neg_OO'] = (merged_df_OO['logreturn_OO'] < merged_df_OO['logreturn_OO'].quantile(0.25)).astype(int)
        merged_df_OO['IP_Pos_CO'] = (merged_df_OO['logreturn_CO'] > merged_df_OO['logreturn_CO'].quantile(0.75)).astype(int)
        merged_df_OO['IP_Neg_CO'] = (merged_df_OO['logreturn_CO'] < merged_df_OO['logreturn_CO'].quantile(0.25)).astype(int)
        
        
        
        
        merged_df_CC['IP_Pos_OC'] = (merged_df_CC['logreturn_OC'] > merged_df_CC['logreturn_OC'].quantile(0.75)).astype(int)
        merged_df_CC['IP_Neg_OC'] = (merged_df_CC['logreturn_OC'] < merged_df_CC['logreturn_OC'].quantile(0.25)).astype(int)  
        merged_df_CC['IP_Pos_CC'] = (merged_df_CC['logreturn_CC'] > merged_df_CC['logreturn_CC'].quantile(0.75)).astype(int)
        merged_df_CC['IP_Neg_CC'] = (merged_df_CC['logreturn_CC'] < merged_df_CC['logreturn_CC'].quantile(0.25)).astype(int)  
        merged_df_CC['IP_Pos_OO'] = (merged_df_CC['logreturn_OO'] > merged_df_CC['logreturn_OO'].quantile(0.75)).astype(int)
        merged_df_CC['IP_Neg_OO'] = (merged_df_CC['logreturn_OO'] < merged_df_CC['logreturn_OO'].quantile(0.25)).astype(int)
        merged_df_CC['IP_Pos_CO'] = (merged_df_CC['logreturn_CO'] > merged_df_CC['logreturn_CO'].quantile(0.75)).astype(int)
        merged_df_CC['IP_Neg_CO'] = (merged_df_CC['logreturn_CO'] < merged_df_CC['logreturn_CO'].quantile(0.25)).astype(int)
        
    
    
    
    # Create the word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=370, relative_scaling=1,
                           normalize_plurals=False, 
                           background_color='white').generate_from_frequencies(word_frequency_dict)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"{ticker}")
    plt.axis('off')
    plt.show()

    return merged_df_OO, bow_OO, merged_df_CC, bow_CC






def RandomForest_BOW(merged_df, bow, IT: str, TF: str):  ##IT stands for impact type {Pos, Neg}, TF stands for timeframe {OC, OO, CC}
    
    df=merged_df.copy()
    XX = merged_df[bow]
    XX = XX.select_dtypes(include=[np.number])
    
    
    
    # Based on the publishing time of headlines response variables are shifted
    if TF in ['OC','OO']:
        df[f'IP_{IT}_{TF}'] = df[f'IP_{IT}_{TF}'].shift(-1) #shifting up response variable to predict next day's movement using current day's data
        df.dropna(subset=[f'IP_{IT}_{TF}'], inplace=True)
    else:
        df[f'IP_{IT}_{TF}'] = df[f'IP_{IT}_{TF}'].shift(-2) #shifting up response variable to predict next day's movement using current day's data
        df.dropna(subset=[f'IP_{IT}_{TF}'], inplace=True)
    
    # 2. Define predictors and target
    X = df[bow]
    # Confirm all columns are numeric
    X = X.select_dtypes(include=[np.number])  # This filters out any accidental object/datetime columns
    
    y = df[f'IP_{IT}_{TF}']
    
    # Adjust the train/test split ratio 👇
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    
    # 3. Build pipeline (scaling helps regularization)
    '''pipeline = Pipeline([
    ('logreg', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000))
    ])'''
    
    
    pipeline = Pipeline([
    ('scaler', StandardScaler()),  # optional, random forests don't *need* scaling, but good practice
    ('rf', RandomForestClassifier(random_state=42))
    ])

    # 4. Define hyperparameter grid
    '''param_grid = {
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'logreg__penalty': ['l1', 'l2']
    }'''
    
    
    param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 5, 10],
    'rf__min_samples_split': [2, 5],
    }
    # 5. Setup cross-validation grid search
    '''grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')'''
    
    grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,           # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1
    )
    try:
        grid_search.fit(X, y)
    except Exception as e:
        print("GridSearch failed:", str(e))
    
    # 6. Best model and parameters
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    
    # 7. Predict probabilities and evaluate
    probs = best_model.predict_proba(X_test)[:, 1]
    #y_pred = best_model.predict(X_test)
    
    
    threshold = 0.25   # 👈 set the custom threshold here
    y_pred = (probs >= threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, probs))
    
    
    # 8. Extract and display coefficients
    '''coef = best_model.named_steps['logreg'].coef_[0]'''
    
    coef = best_model.named_steps['rf'].feature_importances_
    coef_series = pd.Series(coef, index=X.columns)
    
    print("\nTop Positive Predictors:")
    print(coef_series.sort_values(ascending=False).head(10))
    
    print("\nTop Negative Predictors:")
    print(coef_series.sort_values().head(10))
    
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, probs)


    # 9. Get top 5 positive and top 5 negative features
    top_positive = coef_series.sort_values(ascending=False).head(5)
    top_negative = coef_series.sort_values().head(5)


    # 10. Convert to string for summary (e.g., "word1:0.5, word2:0.4")
    top_positive_str = ', '.join([f"{word}:{coef:.2f}" for word, coef in top_positive.items()])
    top_negative_str = ', '.join([f"{word}:{coef:.2f}" for word, coef in top_negative.items()])
    
    # 11. probability of impactful for entire dataset
    predictions = best_model.predict_proba(XX)[:,1]
    
    
    return report, auc, top_positive_str, top_negative_str, predictions
    
 
    
'''
def bench_FIGARCH(data):
    
    df2 = data.copy()
    dataset1=df2[df2['state']=='Negative']
    dataset2=df2[df2['state']=='Positive']

    deg1, loc1, scale1 = t.fit(dataset1['logreturn']) # Negative
    deg2, loc2, scale2 = t.fit(dataset2['logreturn']) # Positive


    # Number of steps to predict
    n_forecast_steps = round(0.1 * len(data))

    # Extract the actual log returns for the forecasted period
    actual_log_returns = np.array(df2["logreturn"][-n_forecast_steps:])
    
    # Storage for predictions
    rolling_forecast_2 = []
    rolling_volatility_2 = []

    # Copy the historical log returns
    historical_returns = df2["logreturn"].copy()
    historical_returns = mstats.winsorize(historical_returns, limits=[wl,wl])
    
    
    # Rolling Forecast Loop
    for i in range(n_forecast_steps):
        # Fit the model to the latest available data
        model2 = arch_model(historical_returns[:-n_forecast_steps + i], vol=vol_type, p=1, o=1, q=1, mean="Constant", dist="normal")
        results2 = model2.fit(disp="off")

        # Forecast next step
        forecast2 = results2.forecast(horizon=1)
        predicted_volatility2 = np.sqrt(forecast2.variance.iloc[-1,0])  # One-step-ahead volatility
        
        #producing the innovation term with Normal dist. and controlling limits for outliers
        rv = np.random.normal(0,1,1)[0]
        while rv > 3 or rv < -3:
            rv = np.random.normal(0,1,1)[0]
        

        # Get the predicted log return (mean) and volatility (std deviation)
        predicted_return2 = predicted_volatility2*rv # One-step-ahead return
        
        # Store results
        rolling_forecast_2.append(predicted_return2)
        rolling_volatility_2.append(predicted_volatility2)



    # Append the predicted return as the next data point for further rolling updates
    #historical_returns = pd.concat([historical_returns, pd.Series([predicted_return])], ignore_index=True)

    # Convert predictions to NumPy arrays
    #rolling_forecast_2 = np.array(mstats.winsorize(np.array(rolling_forecast_2), limits=[0.1,0.1]))
    #rolling_volatility_2 = np.array(mstats.winsorize(np.array(rolling_volatility_2), limits=[0.1,0.1]))

    return actual_log_returns, rolling_forecast_2

'''    
############################################################################################   
    
if __name__ == "__main__":
    
    # Define Excel output path
    script_dir = os.path.dirname(os.path.abspath("AlphaVantage_BOW_dataset.py"))
    output_path = os.path.join(script_dir + "/Alpha_BagOfWords_datasets", 'RF_BOW_summary.xlsx')
    
    
    symbols = ['CRYPTO:BTC', 'CRYPTO:ETH', 'NDAQ', 'NVDA', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA',
               'AVGO', 'COST', 'NFLX', 'WMT', 'JPM', 'V', 'UNH', 'PG', 'JNJ', 'HD', 'KO', 'CRM', 'CVX', 'CSCO', 
               'IBM', 'MRK', 'MCD', 'AXP', 'GS', 'DIS', 'VZ', 'AMGN', 'CAT', 'HON', 'BA', 'NKE', 'SHW', 'MMM',
               'TRV', 'AAPL', 'commodity']
    impact_types = ['Pos', 'Neg']
    time_frames = ['OC', 'OO', 'CO','CC']    
    
    summary_results = {f"RFIP_{IT}_{TF}": [] for IT in impact_types for TF in time_frames}
    
    for symbol in symbols:
        try:
            
            merged_df_OO, bow_OO, merged_df_CC, bow_CC = generate_bow(symbol)
            
            if symbol == 'commodity':
                for IT in impact_types:  
                    report, auc, top_positive_str, top_negative_str, predictions = RandomForest_BOW(merged_df_OO, bow_OO, IT, 'OC')
                    merged_df_OO[f"RFIP_{IT}_OO"] = predictions
                    
                    # Store in results
                    summary_results[f"RFIP_{IT}_OO"].append({
                        'Ticker': symbol,
                        'Accuracy': report['accuracy'],
                        'F1_Positive': report['1.0']['f1-score'],
                        'F1_Negative': report['0.0']['f1-score'],
                        'ROC_AUC': auc,
                        'Top_Positive_Features': top_positive_str,
                        'Top_Negative_Features': top_negative_str
                    })
                    
                merged_df = merged_df_OO
            
            else:
            
                for TF in time_frames:
                    for IT in impact_types:
                        
                        if TF in ['OC','OO']:
                            report, auc, top_positive_str, top_negative_str, predictions = RandomForest_BOW(merged_df_OO, bow_OO, IT, TF)
                            merged_df_OO[f"RFIP_{IT}_{TF}"] = predictions      
                        else:
                            report, auc, top_positive_str, top_negative_str, predictions = RandomForest_BOW(merged_df_CC, bow_CC, IT, TF)
                            merged_df_CC[f"RFIP_{IT}_{TF}"] = predictions
                        
                        # Store in results
                        summary_results[f"RFIP_{IT}_{TF}"].append({
                            'Ticker': symbol,
                            'Accuracy': report['accuracy'],
                            'F1_Positive': report['1.0']['f1-score'],
                            'F1_Negative': report['0.0']['f1-score'],
                            'ROC_AUC': auc,
                            'Top_Positive_Features': top_positive_str,
                            'Top_Negative_Features': top_negative_str
                        })
                    
                merged_df = pd.merge(merged_df_OO, merged_df_CC[['date', 'RFIP_Pos_CC', 'RFIP_Neg_CC', 'RFIP_Pos_CO', 'RFIP_Neg_CO']], on='date', how='outer')
        
            
        except Exception as e:
            print(f"ERROR AS: {e}")
        
        
    # Save summary
    with ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for key, records in summary_results.items():
            if records:  # skip empty
                df = pd.DataFrame(records)
                df.to_excel(writer, sheet_name=key[:31], index=False)  # Excel sheet names max 31 chars
    
    print("Summary saved to RF_BOW_summary.csv")
    
    
    
    
    
    
    
