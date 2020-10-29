import base64
import pickle
import json
from random import choice
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import cv2
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
from table_blocks_det import (
    utils,
    detect_lines_symbols,
    constr_rows,
    constr_blocks,
    detect_table
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

_DTYPE_ITERATIONS_PER_EPOCH = {
    'type_all': 101,
    'type_opl_fact': 84,
    'type_opl': 58
}
_PRETR_MODEL_EPOCHS = {
    'icdar19': 36,
    'icdar13': 1
}
_DEF_TRAINING_EPOCHS = 6
_ITER_STEP = 10
_DETECTION_THRESH = 0.75

_DF_METRICS_PATH = 'results/df.pkl'
_DF_DOCTYPES_PATH = 'results/df_doctypes.pkl'

df = pd.read_pickle(_DF_METRICS_PATH)
df_doctypes = pd.read_pickle(_DF_DOCTYPES_PATH)

def get_log_stat(epoch, dataset_type, pretrained_model, stat):
    log_list = df[(df['pretrained_model'] == pretrained_model) & (df['epoch'] == epoch) & (df['evaluate'] == 'test') & (df['dataset_type'] == dataset_type)]['log'].values[0]

    iter_per_epoch = _DTYPE_ITERATIONS_PER_EPOCH[dataset_type]

    pretr_epochs = _PRETR_MODEL_EPOCHS[pretrained_model]
    total_epochs = pretr_epochs + _DEF_TRAINING_EPOCHS

    iter_start = pretr_epochs * iter_per_epoch
    iter_end = epoch * iter_per_epoch

    final_iter_in_log = (iter_per_epoch // _ITER_STEP) * _ITER_STEP

    stat_list = []
    for log_line in log_list:
        stat_list.append(log_line[stat])
        if log_line['epoch'] == epoch and log_line['iter'] == final_iter_in_log:
            break

    iterations = np.arange(iter_start, iter_end, _ITER_STEP)
    stats = np.array(stat_list)
    return iterations, stats

app.layout = html.Div(children=[
    # Main dashboard:
    dbc.Row([
        # Panels:
        dbc.Col(children=[
            # Titles:
            dbc.Row(
                # A single column with a single Div with info:
                dbc.Col(children=[
                    html.H2('CascadeTabNet finetuning results',
                            style={'text-align': 'center'}),
                    html.Div('Visualizing results of table detection for a set of parameters.')
                ]
                ),
            style={'padding': '15px'}
            ),
            # Pretrained model panel:
            html.Div(children=[
                html.H5('Pretrained model:'),
                dcc.RadioItems(
                    id='pretr_model_radio',
                    options=[
                        {'label': 'ICDAR-19', 'value': 'icdar19'},
                        {'label': 'ICDAR-13', 'value': 'icdar13'}
                    ],
                    value='icdar19',
                    labelStyle={'display': 'block'}
                )
            ],
            style={'padding': '10px',
                   'background-color': '#F8F8FF'}
            ),
            # Dataset type panel:
            html.Div(children=[
                html.H5('Dataset type:'),
                dcc.RadioItems(
                    id='dataset_type_radio',
                    options=[
                        {'label': 'Все типы документов', 'value': 'type_all'},
                        {'label': 'Счета на оплату + счета-фактуры', 'value': 'type_opl_fact'},
                        {'label': 'Счета на оплату', 'value': 'type_opl'}
                    ],
                    value='type_all',
                    labelStyle={'display': 'block'}
                )
            ],
            style={'padding': '10px',
                   'background-color': '#F8F8FF'}
            ),
            # Evaluate model:
            html.Div(children=[
                html.H5('Evaluate on:'),
                dcc.Checklist(
                    id='eval_checklist',
                    options=[
                        {'label': 'train', 'value': 'train'},
                        {'label': 'test', 'value': 'test'},
                        {'label': 'unseen', 'value': 'unseen'}
                    ],
                    value=['unseen'],
                    labelStyle={'display': 'block'}
                )
            ],
            style={'padding': '10px',
                   'background-color': '#F8F8FF'}
            ),
            # Modal:
            html.Div(children=[
                html.H5('Table detection visualization:'),
                dbc.RadioItems(
                    id='random_image_type_radio',
                    options=[
                        {'label': 'train', 'value': 'train'},
                        {'label': 'test', 'value': 'test'},
                        {'label': 'unseen', 'value': 'unseen'}
                    ],
                    inline=True,
                    value='unseen',
                    style={'margin-bottom': '10px'}
                ),
                dbc.Input(
                    id='specify_image_input',
                    placeholder='Specify invoice index...',
                    debounce=True,
                    type='text',
                    style={'margin-bottom': '10px'}
                ),
                dbc.Button('Random/specified image', id='open_image_button'),
                dbc.Modal(
                    [
                        dbc.ModalHeader(
                            id='modal_header'
                        ),
                        dbc.ModalBody(
                            html.Div(
                                id='image'
                            )
                        ),
                        dbc.ModalFooter(
                            dbc.Button('Close', id='close', className='ml-auto')
                        )
                    ],
                    id='modal',
                    size="xl"
                )
            ],
            style={'padding': '15px',
                   'background-color': '#E5F0FF'}
            ),
            # Results:
            html.Div(children=[
                html.H5('Model results:', style={'padding-left': '10px'}),
                html.Div(id='table_results')
            ],
            style={'padding': '10px',
                   'background-color': '#F2F4F6'}
            )
        ],
        width={'size': 3, 'order': 1},
        style={'align-content': 'flex-start'}
        ),
        # Main section
        dbc.Col(children=[
            dcc.Tabs(
                children = [
                    # Tab with training plots, dataset distribution and AP curve
                    dcc.Tab(
                        label='Training plots',
                        children=[
                            # Row with 2 cols: Pie chart and Precision x Recall
                            dbc.Row(children=[
                                dbc.Col(
                                    dcc.Graph(
                                        id='pie_chart'
                                    )
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id='prec_rec'
                                    )
                                )
                            ]),
                            # Row with loss and accuracy plots:
                            dbc.Row(children=[
                                dbc.Col(
                                    dcc.Graph(
                                        id='loss'
                                    )
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id='accuracy'
                                    )
                                )
                            ]),
                            # Row with slider for epoches:
                            dbc.Row(
                                dbc.Col(children=[
                                    dcc.Slider(
                                        id='epoch_slider',
                                        step=None
                                    ),
                                    html.Div('epoches',
                                            style={'textAlign': 'center'})

                                ],
                                width={'size': 7},
                                style={'margin': 'auto'}
                                )
                            )
                        ]
                    ),
                    dcc.Tab(
                        label='Recognition results',
                        children=[
                            html.Div(
                                children=[
                                    html.H5('OCR engine:', style={'padding-left': '10px'}),
                                    dbc.RadioItems(
                                        id='ocr_engine_radio',
                                        options=[
                                            {'label': 'Tesseract', 'value': 'tesseract'},
                                            {'label': 'EasyOCR', 'value': 'easyocr'}
                                        ],
                                        inline=True,
                                        value='tesseract',
                                        style={'margin-bottom': '10px'}
                                    ),
                                    html.H5('Split to textblocks:', style={'padding-left': '10px'}),
                                    dbc.RadioItems(
                                        id='textblocks_radio',
                                        options=[
                                            {'label': 'Not', 'value': 'dont_split'},
                                            {'label': 'Yes', 'value': 'split'}
                                        ],
                                        inline=True,
                                        value='dont_split',
                                        style={'margin-bottom': '10px'}
                                    ),
                                    html.H5('Text to recognize:', style={'padding-left': '10px'}),
                                    dbc.RadioItems(
                                        id='text_radio',
                                        options=[
                                            {'label': 'All text', 'value': 'text_all'},
                                            {'label': 'Without tables', 'value': 'text_nontabular'},
                                            {'label': 'Tables only', 'value': 'text_tabular'}
                                        ],
                                        inline=True,
                                        value='text_all',
                                        style={'margin-bottom': '10px'}
                                    )
                                ],
                                style={'padding': '15px'}
                            ),
                            html.Div(
                                id='text_content',
                                style={
                                    'white-space': 'pre-wrap',
                                    'maxHeight': '700px',
                                    'maxWidth': '1000px',
                                    'overflow': 'scroll'
                                }
                            )
                        ]
                    )
                ]
            )
        ],
        width={'size': 9, 'order': 12}
        )
    ],
    style={'padding': '0px'}
    )

])

@app.callback(
    [Output('epoch_slider', 'min'),
     Output('epoch_slider', 'max'),
     Output('epoch_slider', 'value'),
     Output('epoch_slider', 'marks')],
    [Input('pretr_model_radio', 'value')]
)
def set_epoch_slider(pretrained_model):
    prop_min = _PRETR_MODEL_EPOCHS[pretrained_model]
    prop_max = prop_min + _DEF_TRAINING_EPOCHS
    prop_value = prop_min
    prop_marks = {str(epoch): str(epoch) for epoch in range(prop_min, prop_max + 1)}
    return prop_min, prop_max, prop_value, prop_marks

@app.callback(
    Output('prec_rec', 'figure'),
    [Input('epoch_slider', 'value'),
     Input('eval_checklist', 'value'),
     Input('dataset_type_radio', 'value'),
     Input('pretr_model_radio', 'value')]
)
def update_prec_rec(selected_epoch, eval_list, dataset_type, pretrained_model):
    fig = go.Figure()
    ap_list = []
    for eval_type in eval_list:
        precision = df[(df['pretrained_model'] == pretrained_model) & (df['dataset_type'] == dataset_type) & (df['epoch'] == selected_epoch) & (df['evaluate'] == eval_type)]['precision'].values[0]
        recall = df[(df['pretrained_model'] == pretrained_model) & (df['dataset_type'] == dataset_type) & (df['epoch'] == selected_epoch) & (df['evaluate'] == eval_type)]['recall'].values[0]
        fig.add_trace(
            go.Scatter(x=recall, y=precision, name=eval_type)
        )
        ap = df[(df['pretrained_model'] == pretrained_model) & (df['dataset_type'] == dataset_type) & (df['epoch'] == selected_epoch) & (df['evaluate'] == eval_type)]['ap'].values[0]
        ap_list.append(str(round(ap, 3)))

    fig.update_layout(title={'text': f'Precision x Recall curve <br>' \
                                     f'Epoch {selected_epoch}: {", ".join(eval_list)} <br>' \
                                     f'Average precision: {", ".join(ap_list)}',
                            'x': 0.5,
                            'yanchor': 'top'},
                      xaxis_title='Recall',
                      yaxis_title='Precision',
                      hovermode="x unified")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0.6, 1.05])

    return fig

@app.callback(
    Output('pie_chart', 'figure'),
    [Input('dataset_type_radio', 'value')]
)
def update_pie_chart(dataset_type):
    if dataset_type == 'type_all':
        poss_types = ['opl', 'fact', 'misc']
    elif dataset_type == 'type_opl_fact':
        poss_types = ['opl', 'fact']
    elif dataset_type == 'type_opl':
        poss_types = ['opl']

    map_dict = {'misc': 'Other',
                'opl': 'Payment reports',
                'fact': 'Invoices'}

    poss_df_doctypes = df_doctypes[df_doctypes['img_type'].isin(poss_types)]
    labels = poss_types
    rus_labels = [map_dict[label] for label in labels]
    values = []
    for label in labels:
        values.append(poss_df_doctypes['img_type'].value_counts()[label])

    fig = go.Figure(data=[go.Pie(labels=rus_labels, values=values, textinfo='value+percent')])
    fig.update_layout(
        title={
            'text': 'Document types',
            'x': 0.5,
            'yanchor': 'top',
            'font_size': 25
        },
        legend_font_size = 20,
        font_size = 22,
        annotations=[
            dict(
                x=0.5,
                y=-0.2,
                showarrow=False,
                text='Stratified split: 80% - \\learning data , 20% - \\unseen data <br>Test/train stratified split: 80% / 20%',
                xref='paper',
                yref='paper',
                font = {'size': 14}
            )
        ]
    )
    return fig

@app.callback(
    Output('loss', 'figure'),
    [Input('epoch_slider', 'value'),
     Input('dataset_type_radio', 'value'),
     Input('pretr_model_radio', 'value')]
)
def update_loss(selected_epoch, dataset_type, pretrained_model):
    fig = go.Figure()
    if selected_epoch != 36:
        iterations, loss = get_log_stat(selected_epoch, dataset_type, pretrained_model, 'loss')
        fig = go.Figure(data=go.Scatter(x=iterations, y=loss, name='loss', showlegend=False, hoverinfo='x+y'))
    fig.update_layout(title={'text': 'Loss',
                             'x': 0.5,
                             'yanchor': 'top'},
                      xaxis_title='Iterations',
                      yaxis_title='Loss',
                      hovermode='x')
    iter_per_epoch = _DTYPE_ITERATIONS_PER_EPOCH[dataset_type]

    pretr_epochs = _PRETR_MODEL_EPOCHS[pretrained_model]
    total_epochs = pretr_epochs + _DEF_TRAINING_EPOCHS

    iter_start = pretr_epochs * iter_per_epoch
    iter_end = total_epochs * iter_per_epoch

    fig.update_xaxes(range=[iter_start, iter_end])
    fig.update_yaxes(range=[0, 1.08])

    # Add epochs tracers:
    for num_epoch in range(_DEF_TRAINING_EPOCHS):
        x_epoch = iter_start + (iter_per_epoch * (num_epoch + 1))
        fig.add_trace(
            go.Scatter(
                x=np.array([x_epoch, x_epoch]),
                y=np.array([0, 1]),
                mode='lines',
                line=go.scatter.Line(color="#DF8600", dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )

    return fig

@app.callback(
    Output('accuracy', 'figure'),
    [Input('epoch_slider', 'value'),
     Input('dataset_type_radio', 'value'),
     Input('pretr_model_radio', 'value')]
)
def update_accuracy(selected_epoch, dataset_type, pretrained_model):
    fig = go.Figure()
    if selected_epoch != 36:
        iterations, accuracy = get_log_stat(selected_epoch, dataset_type, pretrained_model, 's0.acc')
        fig = go.Figure(data=go.Scatter(x=iterations, y=accuracy/100, name='Accuracy', showlegend=False, hoverinfo='x+y'))
    fig.update_layout(title={'text': 'Accuracy',
                             'x': 0.5,
                             'yanchor': 'top'},
                      xaxis_title='Iterations',
                      yaxis_title='Accuracy',
                      hovermode='x')
    iter_per_epoch = _DTYPE_ITERATIONS_PER_EPOCH[dataset_type]

    pretr_epochs = _PRETR_MODEL_EPOCHS[pretrained_model]
    total_epochs = pretr_epochs + _DEF_TRAINING_EPOCHS

    iter_start = pretr_epochs * iter_per_epoch
    iter_end = total_epochs * iter_per_epoch

    fig.update_xaxes(range=[iter_start, iter_end])
    fig.update_yaxes(range=[0.91, 1.01])

    # Add asymptote tracer:
    fig.add_trace(
        go.Scatter(
            x=np.array([iter_start, iter_end]),
            y=np.array([1, 1]),
            mode='lines',
            line=go.scatter.Line(color="gray", dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    # Add epochs tracers:
    for num_epoch in range(_DEF_TRAINING_EPOCHS):
        x_epoch = iter_start + (iter_per_epoch * (num_epoch + 1))
        fig.add_trace(
            go.Scatter(
                x=np.array([x_epoch, x_epoch]),
                y=np.array([0, 1]),
                mode='lines',
                line=go.scatter.Line(color="#DF8600", dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    return fig

# Displaying selected image in modal_body Div:
@app.callback(
    [Output('image', 'children'),
     Output('modal_header', 'children')],
    [Input('pretr_model_radio', 'value'),
     Input('dataset_type_radio', 'value'),
     Input('epoch_slider', 'value'),
     Input('random_image_type_radio', 'value'),
     Input('open_image_button', 'n_clicks'),
     Input('specify_image_input', 'value')]
)
def display_modal_image(pretrained_model, dataset_type, selected_epoch, image_type, n_clicks, image_index):
    metrics_folder = f'object_detection_metrics/{pretrained_model}/{dataset_type}/'
    groundtruths_folder = f'{metrics_folder}/groundtruths_{image_type}/'
    detections_folder = f'{metrics_folder}/detections/{selected_epoch}/{image_type}/'

    imagenames_list = f'annotations/COCO_annotations/{dataset_type}/{image_type}_list.json'

    if image_index:
        image_name = f'inv-{str(image_index)}'
    else:
        with open(imagenames_list, 'r+') as f:
            image_names = json.load(f)
        image_name = choice(image_names).split('.')[0]

    modal_header = f'Invoice index: {image_name.split("-")[1]}'

    gth_results_path = f'{groundtruths_folder}{image_name}.txt'
    det_results_path = f'{detections_folder}{image_name}.txt'

    image_path = f'images/{image_name}.jpg'
    img = cv2.imread(image_path)

    # Draw table groundtruths on image:
    with open(gth_results_path, 'r') as f:
        for line in f:
            color = (0, 255, 0)
            _, x1, y1, x2, y2 = line.rstrip().split(' ')
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    # Draw table detections on image:
    with open(det_results_path, 'r') as f:
        for line in f:
            color = (0, 0, 255)
            _, conf, x1, y1, x2, y2 = line.rstrip().split(' ')
            if float(conf) >= _DETECTION_THRESH:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    _, im_arr = cv2.imencode('.jpg', img)
    im_bytes = im_arr.tobytes()
    encoded_image = base64.b64encode(im_bytes)

    # encoded_image = base64.b64encode(open(image_path, 'rb').read())
    contents = 'data:image/png;base64,{}'.format(encoded_image.decode())
    image = html.Img(
        src=contents,
        style={'height': '100%', 'width': '100%'}
    )
    return image, modal_header

@app.callback(
    Output('modal', 'is_open'),
    [Input('open_image_button', 'n_clicks'),
     Input('close', 'n_clicks')],
    [State('modal', 'is_open')]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('table_results', 'children'),
    [Input('pretr_model_radio', 'value'),
     Input('dataset_type_radio', 'value'),
     Input('epoch_slider', 'value')]
)
def update_results_table(pretrained_model, dataset_type, selected_epoch):
    data_list = []
    # Find best model (highest AP_unseen):
    df_unseen = df[df['evaluate'] == 'unseen']
    df_max_ap = df_unseen[(df_unseen['ap'] == df_unseen['ap'].max())]
    df_current = df_unseen[(df_unseen['pretrained_model'] == pretrained_model) & (df_unseen['dataset_type'] == dataset_type) & (df_unseen['epoch'] == selected_epoch)]

    for ind, dtf in enumerate([df_max_ap, df_current]):
        data = dict()
        if ind == 0:
            data['status'] = 'best'
        else:
            data['status'] = 'current'
        data['model'] = f'{dtf["pretrained_model"].values[0]}_' \
                            f'{dtf["dataset_type"].values[0]}_' \
                            f'{dtf["epoch"].values[0]}'
        data['accuracy'] = round(dtf['accuracy'].values[0], 3)
        data['loss'] = round(dtf['loss'].values[0], 3)
        data['AP'] = round(dtf['ap'].values[0], 3)
        data_list.append(data)
    df_results = pd.DataFrame(data_list)
    return dbc.Table.from_dataframe(
        df=df_results,
        hover=True,
        striped=True,
        bordered=True,
        size='sm'
    )

# Display recognized text
@app.callback(
    Output('text_content', 'children'),
    [Input('pretr_model_radio', 'value'),
     Input('dataset_type_radio', 'value'),
     Input('epoch_slider', 'value'),
     Input('random_image_type_radio', 'value'),
     Input('ocr_engine_radio', 'value'),
     Input('textblocks_radio', 'value'),
     Input('text_radio', 'value'),
     Input('specify_image_input', 'value')]
)
def display_recognized_text(pretrained_model, dataset_type, selected_epoch, image_type,
                            ocr_engine, textblocks_split, text_type, image_index):
    if image_index:
        image_name = f'inv-{str(image_index)}'
        image_path = f'images/{image_name}.jpg'
        img = cv2.imread(image_path)

        # Run scripts for textblocks if determined:
        if textblocks_split == 'split':
            detect_lines_symbols.main(image_name, image_path)
            constr_rows.main(image_name, image_path)
            constr_blocks.main(image_name, image_path)
            folder_data = 'table_blocks_det/data'
            text_blocks = utils.load_data(f'{folder_data}/text_blocks_{image_name}.data')

        if text_type == 'text_all':
            with PyTessBaseAPI(lang='rus') as tesseract:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                tesseract.SetImage(image)
                text = tesseract.GetUTF8Text()
            return text

        if text_type == 'text_tabular':
            content = str()
            metrics_folder = f'object_detection_metrics/{pretrained_model}/{dataset_type}/'
            detections_folder = f'{metrics_folder}/detections/{selected_epoch}/{image_type}/'
            det_results_path = f'{detections_folder}{image_name}.txt'
            if textblocks_split == 'dont_split':
                with open(det_results_path, 'r') as f:
                    i = 1
                    for line in f:
                        _, conf, x1, y1, x2, y2 = line.rstrip().split(' ')
                        if float(conf) >= _DETECTION_THRESH:
                            table_img = img[int(y1):int(y2), int(x1):int(x2)]
                            with PyTessBaseAPI(lang='rus') as tesseract:
                                img_rgb = cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(img_rgb)
                                tesseract.SetImage(image)
                                text = f'Table {i}:\n\n' + tesseract.GetUTF8Text()
                                content += text
                                i += 1
            elif textblocks_split == 'split':
                tables = []
                with open(det_results_path, 'r') as f:
                    for line in f:
                        _, conf, x1, y1, x2, y2 = line.rstrip().split(' ')
                        if float(conf) >= _DETECTION_THRESH:
                            table = {'x': int(x1), 'y': int(y1), 'w': int(x2)-int(x1), 'h': int(y2)-int(y1)}
                            tables.append(table)
                detect_table.set_textblocks_table_param(text_blocks, tables)

                for ind_row, row in enumerate(text_blocks):
                    content_row = str()
                    for textblock in row:
                        if textblock['in_table'] == True:
                            x1, x2 = textblock['x'], textblock['x'] + textblock['w']
                            y1, y2 = textblock['y'], textblock['y'] + textblock['h']
                            textblock_img = img[y1:y2, x1:x2]
                            with PyTessBaseAPI(lang='rus', psm=PSM.SINGLE_BLOCK) as tesseract:
                                img_rgb = cv2.cvtColor(textblock_img, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(img_rgb)
                                tesseract.SetImage(image)
                                text = tesseract.GetUTF8Text()
                                content_row += text + '  '
                    content += content_row + '\n'
            return content

        if text_type == 'text_nontabular':
            metrics_folder = f'object_detection_metrics/{pretrained_model}/{dataset_type}/'
            detections_folder = f'{metrics_folder}/detections/{selected_epoch}/{image_type}/'
            det_results_path = f'{detections_folder}{image_name}.txt'
            if textblocks_split == 'dont_split':
                with open(det_results_path, 'r') as f:
                    i = 1
                    for line in f:
                        _, conf, x1, y1, x2, y2 = line.rstrip().split(' ')
                        if float(conf) >= _DETECTION_THRESH:
                            img[int(y1):int(y2), int(x1):int(x2)] = 0

                with PyTessBaseAPI(lang='rus') as tesseract:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(img_rgb)
                    tesseract.SetImage(image)
                    text = tesseract.GetUTF8Text()
                return text

            elif textblocks_split == 'split':
                content = str()
                tables = []
                with open(det_results_path, 'r') as f:
                    for line in f:
                        _, conf, x1, y1, x2, y2 = line.rstrip().split(' ')
                        if float(conf) >= _DETECTION_THRESH:
                            table = {'x': int(x1), 'y': int(y1), 'w': int(x2)-int(x1), 'h': int(y2)-int(y1)}
                            tables.append(table)
                detect_table.set_textblocks_table_param(text_blocks, tables)

                for ind_row, row in enumerate(text_blocks):
                    content_row = str()
                    for textblock in row:
                        if textblock['in_table'] == False:
                            x1, x2 = textblock['x'], textblock['x'] + textblock['w']
                            y1, y2 = textblock['y'], textblock['y'] + textblock['h']
                            textblock_img = img[y1:y2, x1:x2]
                            with PyTessBaseAPI(lang='rus', psm=PSM.SINGLE_BLOCK) as tesseract:
                                img_rgb = cv2.cvtColor(textblock_img, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(img_rgb)
                                tesseract.SetImage(image)
                                text = tesseract.GetUTF8Text()
                                content_row += text + ' '
                    content += content_row + '\n'
                return content

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
