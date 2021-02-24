# img_viewer.py

import PySimpleGUI as sg
import os.path
import re
from PIL import Image, ImageTk
import io
from utils import *

db_img, _, db_names, db_clss = get_hsi()
model = models.load_model('cnn3d_model')
features = models.Model(
    inputs=model.input, outputs=model.layers[-2].output)
db_feature = features.predict(db_img)


def get_img_data(f, maxsize=(494, 768), first=True):
    '''Generate image data using PIL
    '''
    f = f[:-3] + 'bmp'
    img = Image.open(f)
    img.thumbnail(maxsize)
    img = img.resize((63*2,63*2), Image.ANTIALIAS)
    return ImageTk.PhotoImage(img)


folder = 'dataset/AnkaraHSIArchive'

try:
    # Get list of files in folder
    file_list = os.listdir(folder)
except:
    file_list = []

fnames = [
    f
    for f in file_list
    if os.path.isfile(os.path.join(folder, f))
    and f.lower().endswith(('.mat'))
]
fnames = sorted(fnames, key=lambda x: re.search(r'\d+', x).group())

# First the window layout in 2 columns

file_list_column = [
    [
        sg.Listbox(
            values=fnames, enable_events=True, size=(150, 60), key='-FILE LIST-'
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Button('Search', enable_events=True, size=(10, 1), key='-SEARCH-')],
    [sg.Text(size=(40, 1), key='-TOUT-')],
    [sg.Image(key='-IMAGE-')],
    [sg.HSeparator()],
    [sg.Image(key='-R0-'), sg.Image(key='-R1-')],
    [sg.Image(key='-R2-'), sg.Image(key='-R3-')],
    [sg.Image(key='-R4-')],
    [sg.Text(size=(40, 10), key='-SCORE-')],
    [sg.Text(size=(40, 1), key='-T0-')],
    [sg.Text(size=(40, 1), key='-T1-')],
    [sg.Text(size=(40, 1), key='-T2-')],
    [sg.Text(size=(40, 1), key='-T3-')],
    [sg.Text(size=(40, 1), key='-T4-')],



]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window('Hyperspectral CBIR Demo', layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break
    elif event == '-FILE LIST-':  # A file was chosen from the listbox
        filename = os.path.join(folder, values['-FILE LIST-'][0])
        query = filename
        window['-TOUT-'].update(filename)
        window['-IMAGE-'].update(data=get_img_data(filename))
        window['-SEARCH-'].Update(disabled=False)
    elif event == '-SEARCH-':
        query_feature = db_feature[db_names.index(query)]
        closest = k_closest(db_feature, query_feature, norm='l2')
        ac, pr, rc, hl = score(db_clss, closest)
        for i, cl in enumerate(closest[1:]):
            window['-R{}-'.format(i)].update(data=get_img_data(db_names[cl]))
            window['-T{}-'.format(i)].update(db_names[cl].split('/')[-1])
        window['-SCORE-'].update('AC (%): {:.2f}\nPR (%): {:.2f}\nRC (%): {:.2f}\nHL : {:.2f}'.format(
            ac, pr, rc, hl))


window.close()

# 199, 66, 52, 25