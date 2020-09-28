import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

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

_DF_METRICS_PATH = 'results/df_upd.pkl'
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
            style={'padding': '15px',
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
            style={'padding': '15px',
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
            style={'padding': '15px',
                   'background-color': '#F8F8FF'}
            )
        ],
        width={'size': 3, 'order': 1},
        style={'align-content': 'flex-start'}
        ),
        # Visualization
        dbc.Col(children=[
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
                      yaxis_title='Precision')
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

    map_dict = {'misc': 'Другое',
                'opl': 'Счета на оплату',
                'fact': 'Счета-фактуры'}

    poss_df_doctypes = df_doctypes[df_doctypes['img_type'].isin(poss_types)]
    labels = poss_types
    rus_labels = [map_dict[label] for label in labels]
    values = []
    for label in labels:
        values.append(poss_df_doctypes['img_type'].value_counts()[label])

    fig = go.Figure(data=[go.Pie(labels=rus_labels, values=values)])
    fig.update_layout(title={'text': 'Типы документов',
                             'x': 0.5,
                             'yanchor': 'top'})
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
        fig = go.Figure(data=go.Scatter(x=iterations, y=loss))
    fig.update_layout(title={'text': 'Loss',
                             'x': 0.5,
                             'yanchor': 'top'},
                      xaxis_title='Iterations',
                      yaxis_title='Loss')
    iter_per_epoch = _DTYPE_ITERATIONS_PER_EPOCH[dataset_type]

    pretr_epochs = _PRETR_MODEL_EPOCHS[pretrained_model]
    total_epochs = pretr_epochs + _DEF_TRAINING_EPOCHS

    iter_start = pretr_epochs * iter_per_epoch
    iter_end = total_epochs * iter_per_epoch

    fig.update_xaxes(range=[iter_start, iter_end])
    fig.update_yaxes(range=[0, 1.08])
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
        fig = go.Figure(data=go.Scatter(x=iterations, y=accuracy/100))
    fig.update_layout(title={'text': 'Accuracy',
                             'x': 0.5,
                             'yanchor': 'top'},
                      xaxis_title='Iterations',
                      yaxis_title='Accuracy')
    iter_per_epoch = _DTYPE_ITERATIONS_PER_EPOCH[dataset_type]

    pretr_epochs = _PRETR_MODEL_EPOCHS[pretrained_model]
    total_epochs = pretr_epochs + _DEF_TRAINING_EPOCHS

    iter_start = pretr_epochs * iter_per_epoch
    iter_end = total_epochs * iter_per_epoch

    fig.update_xaxes(range=[iter_start, iter_end])
    fig.update_yaxes(range=[0.85, 1.01])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
