![Star Badge](https://img.shields.io/static/v1?label=%F0%9F%8C%9F&message=If%20Useful&style=style=flat&color=BC4E99)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
# Disaster Response Pipeline Project <img src="https://i.pinimg.com/originals/94/3a/77/943a7772c92f036dc059380bc644c05e.gif" align="right" width="150" />
 ## Introduction
 This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

<div align = center>
<img src="img/screenshot.png" align="center" width="800" />

</div>

## ⚙️ Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        
        ```
        python data/process_data.py data disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## 🚧 Folder Structure
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- notebook
|- categories.csv
|- messages.csv
|- ETL Pipeline Preparation.ipynb
|- ML Pipeline Preparation.ipynb
- README.md
```

## 📝 Feedback

If you have any feedback or ideas to improve this project, feel free to contact me via my email at nduongthucanh@gmail.com or:

<a href="https://www.linkedin.com/in/ndleah/">
  <img align="left" alt="Leah's LinkdedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />

</a>
<a href="https://github.com/ndleah">
  <img align="left" alt="Leah's Github" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/github.svg" />
</a>

___________________________________

<p>&copy; 2024 Leah Nguyen</p>
