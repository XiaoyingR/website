from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from datetime import datetime
import json
from decimal import Decimal
from ESquery_init import *
from ESquery import *

global count, category_list, category_choice, date_choice_start, date_choice_end, date_min, date_max, display
count = 0
category_choice = 'All'
RANK = "timeStamp" # choose from 'timeStamp' and 'score'
MAX_SIZE = 200
category_list,  date_min, date_max = ES_Query_init()
category_list.sort()
date_choice_start = date_min
date_choice_end = date_max
# display =  ES_Query(category_choice, date_choice_start, date_choice_end, RANK, MAX_SIZE)


ENTRY_PER_SCROLL = 10

# 显示location——text和location——user
def location_map(): # separate text and user level prediction
    display['loc_user'] = None
    display['loc_text'] = None
    for _ in range(len(display)):
        if display.location[_] == None:
            display['loc_user'][_] = 'N/A'
            display['loc_text'][_] = 'no'
        elif display.location[_].startswith("predict"):
            display['loc_user'][_] = display.location[_][11:]
            display['loc_text'][_] = 'no'
        else:
            display['loc_text'][_] = 'yes'
            display['loc_user'][_] = 'N/A'
    # print(display[['location', 'loc_text']])
    print("done mapping.")


app = Flask(__name__)

@app.route('/')
def display_html():
    global count, category_choice, date_choice_start, date_choice_end, display
    print("Initialization...")
    display = ES_Query(category_choice, date_choice_start, date_choice_end, RANK, MAX_SIZE)

    # convert datetime format
    time_stamp_list = []
    for _ in display['timeStamp']:
        time_stamp_list.append(_[0:10] + ' ' + _[11:16])
    display['timeStamp'] = time_stamp_list
    # change score format
    display['score'] = display['score'].apply(lambda x: str(round(Decimal(x),2)))
    # change caetgory format
    location_map()
    return render_template("index-ver2.html", category=category_list, category_choice=category_choice,\
                           date_start=date_choice_start, date_end=date_choice_end)
    # count = 0
    # return redirect('ajax_request')
    # return redirect(url_for('ajax_request'))


@app.route('/callback', methods=['POST'])
def callback_function():
    global count, category_choice, date_choice_start, date_choice_end, display
    print("I am in route /callback")
    # print(json.loads(request.form.get('data')))
    category_choice = json.loads(request.form.get('data'))["category"]
    date_choice_start_json = json.loads(request.form.get('data'))["date_start"]
    date_choice_end_json = json.loads(request.form.get('data'))["date_end"]

    if category_choice == "":
        category_choice = "All"

    if date_choice_start_json == "":
        date_choice_start = date_min
    else:
            date_choice_start = datetime.strptime(date_choice_start_json[:10],"%d/%m/%Y").date()

    if date_choice_end_json == "":
        date_choice_end = date_max
    else:
        date_choice_end = datetime.strptime(date_choice_end_json[:10],"%d/%m/%Y").date()

    count = 0
    display = ES_Query(category_choice, date_choice_start, date_choice_end, RANK, MAX_SIZE)
    location_map()
    # print("\ntest: ", date_choice_end, category_choice)
    print("\n")
    return redirect(url_for('ajax_request'))
    # return redirect('ajax_request')
    # return redirect("smamiajax")


@app.route('/ajax', methods=['GET', 'POST'])
def ajax_request():
    global count, display
    print("I am in route /ajax ")
    # print("category_choice: ", category_choice)
    # print("date_choice_start: ", date_choice_start)
    # print("date_choice_end", date_choice_end)

    # print(display.head(5))
    length = len(display)
    print("length = ", length)
    try:
        count = json.loads(request.form.get('data'))["value"]
    except:
        count = 0

    print("count = ", count)
    if ((count+1) * ENTRY_PER_SCROLL < length):
        # print(display.head(3))
        data_pass = json.dumps(display.iloc[count*ENTRY_PER_SCROLL:(count+1)*ENTRY_PER_SCROLL].to_dict(orient='list'))
    else: # last update
        index = length - count * ENTRY_PER_SCROLL
        print("In else... index = ", index)
        data_pass = json.dumps(display.iloc[count*ENTRY_PER_SCROLL:count*ENTRY_PER_SCROLL+index].to_dict(orient='list'))
    # print("am I passing data?")
    print(data_pass)
    return data_pass



if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    # when problems occur, try change host to 0.0.0.0
    # app.run(debug=True, port=2400, host="10.0.109.21")
    app.run(port=2400, host="10.0.109.21")
    # app.run(port=2400, host="0.0.0.0")

# research.larc.smu.edu.sg/smami