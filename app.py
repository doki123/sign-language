from contextlib import redirect_stderr
from email.policy import default
from pyexpat import model
from flask_pymongo import PyMongo
from urllib import request
import mediapipe as mp
from json import load
from flask import *
import pandas as pd
import joblib
import numpy as np
import string
import cv2
import time

# loaded_model = joblib.load('fps_handtracking_model.joblib')
loaded_model = joblib.load('world_landmark_hands/world_landmark_model.joblib')


data = {

    'WRIST_X': '',
    'WRIST_Y': '',
    'WRIST_Z': '',

    'THUMB_CMC_X': '',
    'THUMB_CMC_Y': '',
    'THUMB_CMC_Z': '',

    'THUMB_MCP_X': '',
    'THUMB_MCP_Y': '',
    'THUMB_MCP_Z': '',

    'THUMB_IP_X': '',
    'THUMB_IP_Y': '',
    'THUMB_IP_Z': '',

    'THUMB_TIP_X': '',
    'THUMB_TIP_Y': '',
    'THUMB_TIP_Z': '',

    'INDEX_FINGER_MCP_X': '',
    'INDEX_FINGER_MCP_Y': '',
    'INDEX_FINGER_MCP_Z': '',

    'INDEX_FINGER_PIP_X': '',
    'INDEX_FINGER_PIP_Y': '',
    'INDEX_FINGER_PIP_Z': '',

    'INDEX_FINGER_DIP_X': '',
    'INDEX_FINGER_DIP_Y': '',
    'INDEX_FINGER_DIP_Z': '',

    'INDEX_FINGER_TIP_X': '',
    'INDEX_FINGER_TIP_Y': '',
    'INDEX_FINGER_TIP_Z': '',

    'MIDDLE_FINGER_MCP_X': '',
    'MIDDLE_FINGER_MCP_Y': '',
    'MIDDLE_FINGER_MCP_Z': '',

    'MIDDLE_FINGER_PIP_X': '',
    'MIDDLE_FINGER_PIP_Y': '',
    'MIDDLE_FINGER_PIP_Z': '',

    'MIDDLE_FINGER_DIP_X': '',
    'MIDDLE_FINGER_DIP_Y': '',
    'MIDDLE_FINGER_DIP_Z': '',

    'MIDDLE_FINGER_TIP_X': '',
    'MIDDLE_FINGER_TIP_Y': '',
    'MIDDLE_FINGER_TIP_Z': '',

    'RING_FINGER_MCP_X': '',
    'RING_FINGER_MCP_Y': '',
    'RING_FINGER_MCP_Z': '',

    'RING_FINGER_PIP_X': '',
    'RING_FINGER_PIP_Y': '',
    'RING_FINGER_PIP_Z': '',

    'RING_FINGER_DIP_X': '',
    'RING_FINGER_DIP_Y': '',
    'RING_FINGER_DIP_Z': '',

    'RING_FINGER_TIP_X': '',
    'RING_FINGER_TIP_Y': '',
    'RING_FINGER_TIP_Z': '',

    'PINKY_MCP_X': '',
    'PINKY_MCP_Y': '',
    'PINKY_MCP_Z': '',

    'PINKY_PIP_X': '',
    'PINKY_PIP_Y': '',
    'PINKY_PIP_Z': '',

    'PINKY_DIP_X': '',
    'PINKY_DIP_Y': '',
    'PINKY_DIP_Z': '',

    'PINKY_TIP_X': '',
    'PINKY_TIP_Y': '',
    'PINKY_TIP_Z': '',

}

og_data_keys = list(data.keys())
new_data = {}

app = Flask('Jumble', template_folder='/Users/shrutiladiwala/Desktop/Backup/ladiw/Documents/Coding/pycharm_files/GraphsExercisesTests/handtracking_test/teacher_code_fps_timing/multihand_landmark_handtracking/templates')
app.config['MONGO_URI'] = "mongodb://Shruti:bfy6SeOsMbF02Ffp@cluster0-l0gvf.mongodb.net/Shruti_Dats?retryWrites=true&w=majority"
app.config['SECRET_KEY'] = "huh"
mongo = PyMongo(app)


def convert_signs(raw_data):
    for big_index in range(0, 40, 1):
        for small_index in range(0, 21, 1):
            for indiv in ['x', 'y', 'z']:
                # new_data.update({og_data_keys[small_index * ['x', 'y', 'z'].index(indiv)]+str(big_index): raw_data[big_index][small_index][indiv]})
                new_data.update(
                    {og_data_keys[small_index * 3]+str(big_index + 1): raw_data[big_index][small_index]['x']})
                new_data.update(
                    {og_data_keys[small_index * 3 + 1]+str(big_index + 1): raw_data[big_index][small_index]['y']})
                new_data.update(
                    {og_data_keys[small_index * 3 + 2]+str(big_index + 1): raw_data[big_index][small_index]['z']})

    return new_data


def predict_signs(data):
    prediction = loaded_model.predict([list(data.values())])
    strung_chars = string.ascii_lowercase[int(prediction)]
    return strung_chars


# removes issue of 'Unnamed: 0' by treating it as index
model_df = pd.read_csv(
    'world_landmark_hands/world_handtracking_data2.csv', index_col=[0])
print(len(model_df['WRIST_X1']))

# TODO: RECAPTURE DATA IN A SEPARATE CSV AND PLUG THAT IN INSTEAD OF THE CURRENT ONE
# NEW CODE: TRYING TO MAKE A DIFFERENCE BETWEEN MOVING AND NONMOVING SIGNS TO ADD A TRACE
# explanation for future-me: noticed that for j and z signs, it would be convenient to see what motion that hand is making
# ie for the z-sign, it would be nice to actually SEE the z being traced in the air by the index finger
# in order to set that up, current idea is to add a column in the dataset for which finger is moving in each sign
# that way, in the future, we can figure out how to track that finger and trace its movements particularly
# set everything to non-trace except for j and z, which can be hardcoded to track the index tip for now
model_df['MOVING'] = 'NA'
for x in range(0, len(model_df), 1):  # x == index of row

    indiv_class = list(model_df['CLASS'])[x]

    if indiv_class == 9:
        # x is the index of the row, and -1 refers to the MOVING column, as it is the last column in the df
        # here, x would be the index of every row which has a CLASS value of 9 or 25
        # this replaces every MOVING value in that range from NA to INDEX_TIP
        model_df.iloc[[x], -1] = 'PINKY_TIP'

    if indiv_class == 25:
        model_df.iloc[[x], -1] = 'INDEX_TIP'


# END OF NEW CODE


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        return redirect('/')


@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'GET':
        return render_template('add_signs.html')
    else:
        sign_name = request.json['sign_name']
        print(sign_name)
        collected_data = request.json['collected_data']
        converted_data = convert_signs(collected_data)
        converted_data.update({'CLASS': sign_name})

        existing_data = pd.read_csv('world_landmark_hands/world_handtracking_data.csv', index_col=0)
        existing_data = existing_data.append(converted_data, ignore_index=True)
        
        existing_data.to_csv('world_landmark_hands/world_handtracking_data.csv')

        test_see = pd.read_csv('world_landmark_hands/world_handtracking_data.csv', index_col=0)
        print(test_see)
        return redirect('/add')

# def converter(o):
#     if isinstance(o, np.integer):
#         return int(o)


@app.route('/grid', methods=['GET', 'POST'])
def grid():
    if request.method == 'GET':
        return render_template('two_grids_copy.html')
    else:
        example_sign = request.json["example_sign"]
        letters_list = string.ascii_uppercase
        sign_index = letters_list.index(example_sign)

        print(sign_index, example_sign)

        # gets the sign data points for picked letter
        example_sign_dataset = model_df[model_df["CLASS"] == sign_index]
        first_points = example_sign_dataset.iloc[0]

        moving = 0  # whether the sign is moving (aka j and z)
        # this takes what part of the hand is supposed to be tracked to create a trace
        trace_tip = first_points['MOVING']

        if example_sign in ['J', 'Z']:
            moving = 1
            # print(trace_tip)

        # THIS WAS TO AVERAGE OUT THE SIGNS, I FOUND IT BAD FOR MOVING GESTURES
        # averaged_points = example_sign_dataset.mean().values  # averages out the columns to get average value of points
        count = 0

        print(first_points)
        first_points = first_points.drop('MOVING')
        first_points = first_points.astype(float)

        # temporary dictionary that groups up the x-y-zs
        xyz_dictionary = {'x': 0, 'y': 0, 'z': 0}
        # list of the 21 hand points, each point being associated with a dictionary with x, y, z
        grouped_xyz_list = []

        for datapoints in first_points:  # runs through the averaged values
            # each hand point has three columns: x, y, z

            if count == 0:  # gets the x-column datapoint
                xyz_dictionary['x'] = datapoints

            elif count == 1:  # gets the y-column datapoint
                xyz_dictionary['y'] = datapoints

            elif count == 2:  # gets the z-column datapoint
                xyz_dictionary['z'] = datapoints
                grouped_xyz_list.append(xyz_dictionary)
                xyz_dictionary = {'x': 0, 'y': 0, 'z': 0}
                count = -1

            count += 1

        # print(grouped_xyz_list[0:21])

        print(trace_tip)
        return jsonify({'example_datapoints': grouped_xyz_list, 'moving': moving, 'trace_tip': trace_tip})


@app.route('/learn', methods=['GET', 'POST'])
def learn():
    if request.method == 'GET':
        return render_template('learn_sign.html')
    else:
        return redirect('/learn')


@app.route('/practice', methods=['GET', 'POST'])
def practice():
    if request.method == 'GET':
        return render_template('practice_signs.html')
    else:
        picked_letter = request.json["picked_sign"]
        raw_data = request.json["raw_data"]
        converted_data = convert_signs(raw_data)
        strung_chars = predict_signs(converted_data)
        print(picked_letter)
        print(strung_chars)
        if picked_letter == strung_chars.upper():
            result = "Correct; " + strung_chars.upper()
        else:
            result = "Incorrect; " + strung_chars.upper()

        return jsonify({'prediction': result})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # # print(request.json)  # this is the current data taken from the HTML --> put everything in one long list and predict
        raw_data = request.json

        # # has 40 dictionaries for 40 frames, each dictionary has 21 dictionaries (thumb, wrist, etc) each with x-y-x coords
        # # data from fps_handtracking has the thumb-wrist-etc all in one big dictionary
        # # merge the smaller 21 dictionaries --> one large dictionary
        # # print(raw_data[0][0])
        # # raw_test.update(
        # #     {'WRIST_X1': raw_data[0][0]['x'], 'WRIST_Y1': raw_data[0][0]['y'], 'WRIST_z1': raw_data[0][0]['z']})
        # # print(raw_test)
        # for big_index in range(0, 40, 1):
        #     for small_index in range(0, 21, 1):
        #         for indiv in ['x', 'y', 'z']:
        #             # new_data.update({og_data_keys[small_index * ['x', 'y', 'z'].index(indiv)]+str(big_index): raw_data[big_index][small_index][indiv]})
        #             new_data.update({og_data_keys[small_index * 3]+str(big_index): raw_data[big_index][small_index]['x']})
        #             new_data.update({og_data_keys[small_index * 3 + 1]+str(big_index): raw_data[big_index][small_index]['y']})
        #             # NOTE: in the og dataset, Thumb_IP_Z + THUMB_TIP_Z wwere overwritten --> when a new dataset is created, and the dataset is 2520, remove the if
        #             if og_data_keys[small_index * 3 + 2] in ['THUMB_IP_Z', 'THUMB_TIP_Z']:
        #                 new_data.update({og_data_keys[small_index * 3 + 2]: raw_data[big_index][small_index]['z']})
        #             else:
        #                 new_data.update({og_data_keys[small_index * 3 + 2]+str(big_index): raw_data[big_index][small_index]['z']})

        # prediction = loaded_model.predict([list(new_data.values())])
        # strung_chars = string.ascii_lowercase[int(prediction)]
        # print(strung_chars)

        converted_data = convert_signs(raw_data)
        strung_chars = predict_signs(converted_data)
        # strung_chars = predict_signs(raw_data)

        return jsonify({'success': True, 'prediction': strung_chars})


if '__main__' == __name__:
    app.run(debug=True)
