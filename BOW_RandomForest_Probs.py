"""
Forecasting Asset Returns with Sentiment-Enhanced FIGARCH Models
Author: Mostafa Abdolahi Moghadam

Description:
This script processes AlphaVantage financial news headlines and financial return data.
It performs NLP preprocessing (tokenization, lemmatization, custom stopwords) to create 
a Bag-of-Words (TF-IDF) feature set. It then trains a Random Forest classifier using 
cross-validation to predict the probability of a news headline significantly impacting 
asset returns across various intraday time frames (OC, OO, CC, CO). 
"""

import gc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pandas import ExcelWriter

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Free up memory
gc.collect()


def Window_separation(df): 
    """
    Aligns news headlines with specific market trading windows based on their publication time.
    Standard market hours are defined as 9:30 AM to 4:00 PM.
    """
    df = df.dropna(subset=['time_published']).copy()
    
    # Parse time_published into a full datetime column
    df['datetime_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
    
    # Extract date and time separately
    df['date'] = df['datetime_published'].dt.date
    df['time'] = df['datetime_published'].dt.time
    
    # Define market open and close times
    market_open_hour, market_open_minute = 9, 30
    market_close_hour, market_close_minute = 16, 0
    
    def get_window_Open_datetime(row):
        """Calculate start of window for each headline (previous day's market open)."""
        dt = row['datetime_published']
        market_open_today = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                                         hour=market_open_hour, minute=market_open_minute)
        if dt >= market_open_today:
            return market_open_today
        else:
            return market_open_today - pd.Timedelta(days=1)
    
    def get_window_Close_datetime(row):
        """Calculate end of window for each headline (next day's market close)."""
        dt = row['datetime_published']
        market_close_today = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                                         hour=market_close_hour, minute=market_close_minute)
        if dt <= market_close_today:
            return market_close_today
        else:
            return market_close_today + pd.Timedelta(days=1)
    
    # Calculate window start and end for each headline
    df['window_start'] = df.apply(get_window_Open_datetime, axis=1)
    df['window_end'] = df.apply(get_window_Close_datetime, axis=1)
    
    # Group headlines by window_start and window_end date
    df['window_Open_to_Open'] = df['window_start'].dt.date
    df['window_Close_to_Close'] = df['window_end'].dt.date
    
    return df


def generate_bow(ticker):
    """
    Reads dataset, performs NLP preprocessing, creates TF-IDF features, 
    and defines binary target variables based on return quantiles.
    """
    # Use relative paths assuming data is in a 'data' folder in the repo
    df_raw = pd.read_csv(f"./Datasets/Alpha_FinBERT_datasets/{ticker}_Alpha_FinBERT_dataset.csv")
    df = Window_separation(df_raw)

    # Import frequency-based dataset to merge with BOW
    df2 = pd.read_csv(f"./data/Frequency-based_Datasets/{ticker}_freq_dataset.csv")
    df2['date'] = pd.to_datetime(df2['date'])
    
    text = df['title'].tolist()
    
    # Load spaCy English model (Requires running: python -m spacy download en_core_web_sm)
    nlp = spacy.load('en_core_web_sm')
    
    def get_wordnet_pos_spacy(spacy_pos):
        """Map spaCy POS tag to WordNet POS tag."""
        if spacy_pos.startswith('J'): return wordnet.ADJ
        elif spacy_pos.startswith('N'): return wordnet.NOUN
        elif spacy_pos.startswith('V'): return wordnet.VERB
        elif spacy_pos.startswith('R'): return wordnet.ADV
        else: return wordnet.NOUN
    
    # Custom stopwords setup
    custom_stopwords = set(stopwords.words('english'))
    additional_stopwords = {".com","cryptos","crypto","cryptocurrency","$","in", "for", "etc.", 
                            "'s", "one", " ", 'e', 'g', 'u', 'wo', 'take', 'get', 'make', 
                            'back', 'like', 'another'}
    stopwords_list = list(custom_stopwords.union(additional_stopwords))
    
    lemmatizer = WordNetLemmatizer()
    
    # Financial domain synonyms mapping
    synonym_dict = {
        'american': 'usa', 'us': 'usa', 'sp': 'sp500', 's&p': 'sp500',
        'alphabet': 'google', 'employee':'worker', 'china': 'chinese', 
        'firm': 'company', 'covid19': 'coronavirus', 'tech': 'technology',
        'wall street': 'wall_street', 'donald': 'donald trump', 
        'apple_inc.': 'apple', 'warren': 'BerkshireHathaway', 
        'bitcoin': 'crypto', 'dow': 'DowJonesIndex', 'grow': 'growth'
    }
    
    def custom_tokenizer(doc):
        """Tokenize, remove stopwords/punctuation, lemmatize, and apply synonyms."""
        spacy_doc = nlp(doc)
        tokens = []
        for ent in spacy_doc.ents:
            ent_token = ent.text.replace(' ', '_').lower()
            if ent_token not in stopwords_list:
                tokens.append(ent_token)
    
        for token in spacy_doc:
            token_text = token.text.lower()
            if (token.ent_type_ == '' and not token.is_punct and token_text not in stopwords_list):
                wn_pos = get_wordnet_pos_spacy(token.pos_)
                lemma = lemmatizer.lemmatize(token_text, wn_pos)
                replaced = synonym_dict.get(lemma, lemma)
                if replaced not in stopwords_list:
                    tokens.append(replaced)
        return tokens
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=300, tokenizer=custom_tokenizer)
    bow_matrix = vectorizer.fit_transform(text)
    feature_names = vectorizer.get_feature_names_out()
    word_frequencies = bow_matrix.sum(axis=0).tolist()[0]
    
    word_frequency_dict = dict(zip(feature_names, word_frequencies))
    
    # Create DataFrames of the Bag of Words matrix
    bow_df_OO = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)
    bow_df_OO = bow_df_OO.drop(columns=[col for col in bow_df_OO.columns if col in list(df2.columns)])
    bow_df_CC = bow_df_OO.copy()
    
    bow_df_OO['date'] = pd.to_datetime(df['window_Open_to_Open'].copy())
    bow_OO = bow_df_OO.columns
 
    bow_df_CC['date'] = pd.to_datetime(df['window_Close_to_Close'].copy())
    bow_CC = bow_df_CC.columns
    
    # Group by date and sum the word counts 
    daily_bow_df_OO = bow_df_OO.groupby('date').sum().reset_index()
    daily_bow_df_CC = bow_df_CC.groupby('date').sum().reset_index()
    
    # Merge datasets
    merged_df_OO = pd.merge(daily_bow_df_OO, df2, on='date', how='inner', suffixes=('', '_df2'))
    merged_df_CC = pd.merge(daily_bow_df_CC, df2, on='date', how='inner', suffixes=('', '_df2'))
    
    # Create binary response variables based on 75th (Positive) and 25th (Negative) percentiles
    if ticker == 'commodity':
        for df_subset in [merged_df_OO, merged_df_CC]:
            df_subset['IP_Pos_OC'] = (df_subset['logreturn'] > df_subset['logreturn'].quantile(0.75)).astype(int)
            df_subset['IP_Neg_OC'] = (df_subset['logreturn'] < df_subset['logreturn'].quantile(0.25)).astype(int)
    else:
        for df_subset in [merged_df_OO, merged_df_CC]:
            df_subset['IP_Pos_OC'] = (df_subset['logreturn_OC'] > df_subset['logreturn_OC'].quantile(0.75)).astype(int)
            df_subset['IP_Neg_OC'] = (df_subset['logreturn_OC'] < df_subset['logreturn_OC'].quantile(0.25)).astype(int)  
            df_subset['IP_Pos_CC'] = (df_subset['logreturn_CC'] > df_subset['logreturn_CC'].quantile(0.75)).astype(int)
            df_subset['IP_Neg_CC'] = (df_subset['logreturn_CC'] < df_subset['logreturn_CC'].quantile(0.25)).astype(int)  
            df_subset['IP_Pos_OO'] = (df_subset['logreturn_OO'] > df_subset['logreturn_OO'].quantile(0.75)).astype(int)
            df_subset['IP_Neg_OO'] = (df_subset['logreturn_OO'] < df_subset['logreturn_OO'].quantile(0.25)).astype(int)
            df_subset['IP_Pos_CO'] = (df_subset['logreturn_CO'] > df_subset['logreturn_CO'].quantile(0.75)).astype(int)
            df_subset['IP_Neg_CO'] = (df_subset['logreturn_CO'] < df_subset['logreturn_CO'].quantile(0.25)).astype(int)
            
    # Generate and display Word Cloud
    wordcloud = WordCloud(width=800, height=400, max_words=370, relative_scaling=1,
                          background_color='white').generate_from_frequencies(word_frequency_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"{ticker} Word Cloud")
    plt.axis('off')
    plt.show()

    return merged_df_OO, bow_OO, merged_df_CC, bow_CC


def RandomForest_BOW(merged_df, bow, IT: str, TF: str):  
    """
    Trains a Random Forest classifier using GridSearch cross-validation.
    IT stands for impact type {Pos, Neg}, TF stands for timeframe {OC, OO, CC, CO}.
    """
    df = merged_df.copy()
    XX = merged_df[bow].select_dtypes(include=[np.number])
    
    # Shift response variable to predict next day's movement using current day's data
    shift_val = -1 if TF in ['OC','OO'] else -2
    df[f'IP_{IT}_{TF}'] = df[f'IP_{IT}_{TF}'].shift(shift_val) 
    df.dropna(subset=[f'IP_{IT}_{TF}'], inplace=True)
    
    # Define predictors and target
    X = df[bow].select_dtypes(include=[np.number]) 
    y = df[f'IP_{IT}_{TF}']
    
    # Train/Test Split (Default 80/20 setup from robustness checks)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Build Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Define hyperparameter grid
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 5, 10],
        'rf__min_samples_split': [2, 5],
    }
    
    # Cross-validation grid search (Strictly on Training Data to prevent leakage)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    try:
        grid_search.fit(X_train, y_train)  # Fixed: Fit only on training set
    except Exception as e:
        print("GridSearch failed:", str(e))
    
    best_model = grid_search.best_estimator_
    
    # Predict probabilities on test set
    probs = best_model.predict_proba(X_test)[:, 1]
    
    # Custom prediction threshold (Base configuration: 0.25)
    threshold = 0.25   
    y_pred = (probs >= threshold).astype(int)
    
    # Extract and sort feature importance coefficients
    coef = best_model.named_steps['rf'].feature_importances_
    coef_series = pd.Series(coef, index=X.columns)
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    # Get top positive/negative predictors as strings for output summary
    top_positive = coef_series.sort_values(ascending=False).head(5)
    top_negative = coef_series.sort_values().head(5)
    top_positive_str = ', '.join([f"{word}:{coef:.2f}" for word, coef in top_positive.items()])
    top_negative_str = ', '.join([f"{word}:{coef:.2f}" for word, coef in top_negative.items()])
    
    # Generate probabilities for the entire dataset to pass back
    predictions = best_model.predict_proba(XX)[:,1]
    
    return report, auc, top_positive_str, top_negative_str, predictions
    
    
if __name__ == "__main__":
    
    # Define relative output path
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'RF_BOW_summary.xlsx')
    
    symbols = ['CRYPTO:BTC', 'CRYPTO:ETH', 'NDAQ', 'NVDA', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA',
               'AVGO', 'COST', 'NFLX', 'WMT', 'JPM', 'V', 'UNH', 'PG', 'JNJ', 'HD', 'KO', 'CRM', 'CVX', 'CSCO', 
               'IBM', 'MRK', 'MCD', 'AXP', 'GS', 'DIS', 'VZ', 'AMGN', 'CAT', 'HON', 'BA', 'NKE', 'SHW', 'MMM',
               'TRV', 'AAPL', 'commodity']
    impact_types = ['Pos', 'Neg']
    time_frames = ['OC', 'OO', 'CO','CC']    
    
    summary_results = {f"RFIP_{IT}_{TF}": [] for IT in impact_types for TF in time_frames}
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        try:
            merged_df_OO, bow_OO, merged_df_CC, bow_CC = generate_bow(symbol)
            
            if symbol == 'commodity':
                for IT in impact_types:  
                    report, auc, top_pos, top_neg, predictions = RandomForest_BOW(merged_df_OO, bow_OO, IT, 'OC')
                    merged_df_OO[f"RFIP_{IT}_OO"] = predictions
                    
                    summary_results[f"RFIP_{IT}_OO"].append({
                        'Ticker': symbol,
                        'Accuracy': report['accuracy'],
                        'F1_Positive': report.get('1', report.get('1.0', {})).get('f1-score', 0),
                        'F1_Negative': report.get('0', report.get('0.0', {})).get('f1-score', 0),
                        'ROC_AUC': auc,
                        'Top_Positive_Features': top_pos,
                        'Top_Negative_Features': top_neg
                    })
            else:
                for TF in time_frames:
                    for IT in impact_types:
                        if TF in ['OC','OO']:
                            report, auc, top_pos, top_neg, predictions = RandomForest_BOW(merged_df_OO, bow_OO, IT, TF)
                        else:
                            report, auc, top_pos, top_neg, predictions = RandomForest_BOW(merged_df_CC, bow_CC, IT, TF)
                        
                        summary_results[f"RFIP_{IT}_{TF}"].append({
                            'Ticker': symbol,
                            'Accuracy': report['accuracy'],
                            'F1_Positive': report.get('1', report.get('1.0', {})).get('f1-score', 0),
                            'F1_Negative': report.get('0', report.get('0.0', {})).get('f1-score', 0),
                            'ROC_AUC': auc,
                            'Top_Positive_Features': top_pos,
                            'Top_Negative_Features': top_neg
                        })
        except Exception as e:
            print(f"ERROR processing {symbol}: {e}")
        
    # Save summary to Excel
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for key, records in summary_results.items():
            if records:  
                df_out = pd.DataFrame(records)
                df_out.to_excel(writer, sheet_name=key[:31], index=False)  
    
    print(f"Summary successfully saved to {output_path}")
