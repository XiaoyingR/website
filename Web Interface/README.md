## Web Interface
The interface aims at visualizing the records in database. While framework is built on Flask, Template is designed by Jessica.


### A Screenshot
![screenshot](https://github.com/LARC-CMU-SMU/SMAMI/blob/master/Web%20Interface/screeenshot.PNG)

### Backend Components
* **ESquery_init** - it initializes the category components and get the max & min dates
* **Esquery** - search ES with predefined criteria and return a pandas dataframe
* **main_page** - Flask framework to render the webpage

### Frontend Components
In addtion to main search on date and category, there are:
* a scroll-down button to show LARC's web pages
* automatic scroll down
* a go-to-top button