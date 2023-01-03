# The Disaster response project for Udacity cource

## Motivation
The purpose of the project is to build a model for an API that classifies disaster messages based on data from Figure Eight.


## File structure
    .
    ├── analyse data                #Folder with notebooks to analys.
    │   ├── ETL Pipeline Preparation.ipynb      #ETL analyse
    │   ├── ML Pipeline Preparation.ipynb       #ML analyse
    ├── app                         
    │   ├── template                
    |   │   ├── go.html             # classification result page of web app
    |   │   ├── master.html         # main page of web app
    │   ├── run.py                  # Flask file that runs app
    ├── data                        
    │   ├── InsertDatabaseName.db   # database to save clean data to
    │   ├── process_data.py         # ETL pipeline
    ├── models                      
    │   ├── classifier.pkl          # saved model 
    │   ├── train_classifier.py     # Model pipeline
    ├── README.md                   # This file
    └── requirements.txt            # File for reproducing the enviroment.






## Install
This project depends on Python 3.x and needs the libraries that are described in the requirements.txt

Tot prepare the enviroment execute the code below:
```shell
pip install -U requirements.txt
```


## Run project
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

## License


### The data files associated with this project are from Figure Eight
- messages.csv: FIgure Eight provide x messages
- categories.csv: Raw categories data, total x categories.



# Check before sending the assignment:
https://review.udacity.com/#!/rubrics/1565/view



