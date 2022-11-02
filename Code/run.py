import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

#from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'scikit-learn'])

#Updated to include additional chart showing top categories

app = Flask(__name__)

def tokenize(text):
    '''
    Create tokens and then lemmatize the tokens. Useful for including in CountVectorizer
    Parameters:
        text (str): Input text to be tokenized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#engine = create_engine('sqlite:///../data/YourDatabaseName.db')
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
#engine = create_engine('sqlite:////home/prasannaiyer/Projects/NLP_Project/Data/DisasterResponse.db')
db_filepath = './Data/DisasterResponse1.db'
db_filepath_sql = 'sqlite:///' + db_filepath
engine = create_engine(db_filepath_sql)
df = pd.read_sql_table('Message_Category', engine)

# load model
#model = joblib.load("../models/your_model_name.pkl")
#model = joblib.load("../models/classifier.pkl")
#model = pickle.load(open('../models/classifier.pkl', 'rb'))
## Load the model file ##
model_filepath = './Code/cv_model1.sav'
model = pickle.load(open(model_filepath, 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_category = df.melt(id_vars= ['id', 'genre', 'message', 'original'], value_vars= df.columns[~df.columns.isin(['id', 'genre', 'message', 'original'])])
    top_category_values = list(df_category.groupby(by = 'variable')['value'].sum().sort_values(ascending = False)[:10].values)
    top_category_names = list(df_category.groupby(by = 'variable')['value'].sum().sort_values(ascending = False)[:10].index)

    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres -1',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_values
                )
            ],

            'layout': {
                'title': 'Top Categories by Occurence',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

def chart_top_categories():
    
    # extract data needed for visuals
    print('HERE') 
    df_category = df.melt(id_vars= ['id', 'genre', 'message', 'original'], value_vars= df.columns[~df.columns.isin(['id', 'genre', 'message', 'original'])])
    top_category_values = list(df_category.groupby(by = 'variable')['value'].sum().sort_values(ascending = False)[:10].values)
    top_category_names = list(df_category.groupby(by = 'variable')['value'].sum().sort_values(ascending = False)[:10].index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()