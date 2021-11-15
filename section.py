import dash_html_components as html

import conteneur

import dash_core_components as dcc

# def section_00():
#     section = html.Section(
#         className="section_content",
#         children=[
#             html.Div(
#                 children=[
#                     html.H3('Cover page', id="partie00", className="titre_menu"),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#                     html.Br(), html.Br(),
#         ])])
#     return section

def section_01():
    section = html.Section(
        className="section_content",
        children=[
            html.Div(
                children=[
                    html.H3('Input data', id="partie01", className="titre_menu"),
                    html.A(html.Button('Refresh Data'), href='/'),
                    html.Br(), html.Br(),
                    html.P("To launch the application, the following data files are required :"),
                    html.Li("KPI data : hourly indicators on analyzed cells (RTWP, Avg Num HSUPA Users, ...),"),
                    html.Li(
                        "Localization data : informations about the position of all cells in the first data file (Latitude, Longitude),"),
                    html.Li(
                        "Tier data : gives the neighbor cells for all analyzed cells."),
                    html.Br(), html.Br(),
                    html.Div([
                        html.H6('KPI data'),# style = {'margin-left' : '180px'}),
                        dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                         style={
                        #     'width': '60%',
                        #     'height': '60px',
                             'lineHeight': '60px',
                             'borderWidth': '1px',
                             'borderStyle': 'dashed',
                             'borderRadius': '50px',
                             'textAlign': 'center',
                        #     'margin-left': '100px'
                         },
                        # Allow multiple files to be uploaded
                        multiple=True
                    ),

                        #html.Div(id='output-data-upload')
                    ], style={'width': '33%', 'display': 'inline-block', 'margin-bottom' : '50px', 'textAlign' : 'center'}),
                    html.Div([
                        html.H6('Localization data'),# style = {'margin-left' : '150px'}),
                        dcc.Upload(
                            id='upload-data_loc',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                            #     'width': '60%',
                            #     'height': '60px',
                                 'lineHeight': '60px',
                                 'borderWidth': '1px',
                                 'borderStyle': 'dashed',
                                 'borderRadius': '50px',
                                 'textAlign': 'center',
                            #     'margin-left': '100px'
                             },
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),

                        #html.Div(id='output-data_loc-upload')
                    ],
                        style={'width': '33%', 'display': 'inline-block', 'margin-bottom' : '50px', 'textAlign' : 'center'}),
                    html.Div([
                        html.H6('Tier data'),# style={'margin-left': '180px'}),
                        dcc.Upload(
                            id='upload-data_crown',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                             style={
                            #    'width': '60%',
                            #     'height': '60px',
                                 'lineHeight': '60px',
                                 'borderWidth': '1px',
                                 'borderStyle': 'dashed',
                                 'borderRadius': '50px',
                                 'textAlign': 'center',
                            #     'margin-left': '100px'
                             },
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),

                        # html.Div(id='output-data_loc-upload')
                    ],
                        style={'width': '33%', 'display': 'inline-block', 'margin-bottom': '50px', 'textAlign' : 'center'})
                ]),

            html.Div([
                 html.Div(id='output-data-upload', style = {'width': '24%', 'display': 'inline-block', 'textAlign' : 'center'}),
                 html.Div(id='output-data_loc-upload', style={'width': '37%', 'display': 'inline-block', 'textAlign' : 'center'}),
                 html.Div(id='output-data_crown-upload', style={'width': '33%', 'display': 'inline-block', 'textAlign' : 'center'})

            ])

        ]
    )
    return section

def section_0():
    """
    Création d'une section
    Les differentes classe CSS :
        - "titre_section" à affeter au titre de la section

    Pensez à modifier l'id du titre de la section pour qu'il coresponde au paramètre href du menu
    """
    section = html.Section(
        className="section_content",
        children=[
            html.H3('Detection course parameters', id="partie0", className="titre_menu"),
            html.Br(),
            html.P("Please set up the parameters or keep the default values (recommended)."),

            html.Br(), #html.Br(),
            html.Div(children = [
                html.Div(
                    style={'width': '50%', 'display': 'inline-block'},
                    children = [
            #html.H4('Required parameters'),
            html.H6('Data preparation'),
            html.Li("In the data preparation phase, the algorithm removes straightforward anomalies to concentrate on anomalies that are harder to detect."),
            html.P("For this reason, cells with an RTWP value higher than the following threshold are considered as obvious anomalies."),
            html.P("Please set-up this threshold. The default (recommended) value is -85 dBm."),
            dcc.Input(placeholder = 'Max RTWP...',
                      id = 'input-max_rtwp', type = 'number', step = 1, value = -85),
            html.Br(), html.Br(),
            html.Li("If the proportion of missing data exceeds the following threshold for a given cell, then, no analysis is performed for this cell."),
            html.P("Please set up this threshold. The default (recommended) value is 30%."),
            dcc.Input(placeholder = '% missing data...',
                      id = 'input-miss_data', type = 'number', step = 1, value = 30),
            html.Br(), html.Br(),
            html.H6('Anomaly detection'),
            html.Li("Three levels of detection are pre-defined based on the impact of the interference :"),
            html.P("Hard detection : in this case, only highly impacting interferences are detected,"),
            html.P("Soft detection : in this case, interference cases that cause minor impact are also identified,"),
            html.P("Average detection : this scenario is an average level in between the two former ones."),
            dcc.Dropdown(id='dropdown_detection_level',
                          options=[{'label' : 'Soft detection', 'value':'Soft'}, {'label' : 'Average detection', 'value':'Average'}, {'label' : 'Hard detection', 'value':'Hard'}],
                          searchable = False,
                          placeholder = 'Detection level...',
                          style={
                              'width': '60%'},
                              value = 'Average'),
            html.Br(),

            ]),

                html.Div(
                    style={'width': '5%', 'display': 'inline-block'},
                    children=[html.Br()]),
                html.Div(
                    style={'width': '45%', 'display': 'inline-block'},
                children = [
                    html.H6('Detection course output'),
                    html.Li("The following KPIs can be considered for the analysis and visualisation of the detection course results."),
                    html.P("The selected KPIs will be combined to define a QoS degradation metric."),
                    html.P("This QoS degradation will be visualized and used for cells prioritization."),
                    html.P("Please select the KPIs to be used for QoS degradation estimation."),
                    dcc.Checklist(
                        id = 'checklist_qos',
                        options=[
                            {'label': 'DCR-PS', 'value': 'DCR-PS'},
                            {'label': 'CSSR-CS', 'value': 'CSSR-CS'},
                            {'label': 'CSSR-PS', 'value': 'CSSR-PS'}
                            ],
                        value=['DCR-PS','CSSR-CS','CSSR-PS'],
                        labelStyle={'display': 'inline-block'}
                        ),
                    html.Br(),
                    html.Li("The QoS degradation metric is a weighted average of the selected KPIs."),
                    html.P("Please define the weights of each KPI."),
                    html.I("Note that if a KPI is not selected, its weight is ignored."),
                    html.Br(),
                    dcc.Input(placeholder='DCR-PS...',
                              id='input-DCR-PS', type='number', min=1, max=10, step=1, value=1),
                    dcc.Input(placeholder='CSSR-CS...',
                              id='input-CSSR-CS', type='number', min=1, max=10, step=1, value=1),
                    dcc.Input(placeholder='CSSR-PS...',
                              id='input-CSSR-PS', type='number', min=1, max=10, step=1, value=1),
                    html.Br(), html.Br(),
                    html.Li("The detection course may detect a large number of anomalies."),
                    html.P("Those anomalies are ranked in a way that you can see first the most relevant ones."),
                    html.P("Please choose the criteria that will be considered for this ranking."),
                    dcc.Checklist(
                        id = 'checklist_prio',
                        options=[
                            {'label': 'Last anomaly', 'value': 'last anomaly'},
                            {'label': 'Anomaly duration', 'value': 'anomaly duration'},
                            {'label': 'QoS degradation', 'value': 'qos degradation'},
                            {'label': 'Maximum RTWP', 'value': 'max RTWP'}
                            ],
                        value=['last anomaly','anomaly duration','qos degradation','max RTWP'],
                        labelStyle={'display': 'inline-block'}
                        ),
                    html.Br(), html.Br(),
            #     html.H4('Optional parameters'),
            #     html.H6('Pre-clustering'),

                    html.Br(), html.Br(),
                    html.Br(),

                ])
                    ]),
            html.Br(),
            html.Br(),
            
            html.Button(id='graph_button_0', children='Process anomaly detection', n_clicks=0),
            html.Br(), html.Br(),
            html.I("The detection course may take some time processing. "),
            html.B("It can last 1h-1h30."),
            html.Br(), html.Br(),
            html.Div(id='detection_course_status_start'),
            html.Div(id='detection_course_status_end'),

        ])
    return section

def section_1():
    """
    Création d'une section
    Les differentes classe CSS :
        - "titre_section" à affeter au titre de la section

    Pensez à modifier l'id du titre de la section pour qu'il coresponde au paramètre href du menu
    """
    section = html.Section(
        className="section_content",
        children=[
            html.H3('Clustering', id="partie1", className="titre_menu"),
            html.P("The objective here is to group cells that are similar in terms of KPIs behaviour and that are in the same geographical area and environment."),
            html.P(" For this purpose:"),
            html.Li("We first define groups of cells that correspond to same environment (areas with homogeneous site density), as shown in the left side graph. Those groups are called “localization pre-clusters”,"),
            html.Li("Then, each localization pre-cluster is split into homogeneous sub-groups in terms of KPIS behaviour. A localization pre-cluster i, can be split into K pre-clusters i,j, j going from 1 to K. For each of those pre-clusters, anomaly thresholds for each KPI are determined. "
                    "Pre-clusters as well as their corresponding thresholds are shown in the following right hand table."),
                            
        conteneur.conteneur_2_graph(["graph_preclust", "table_preclust"])
                            ,
                            html.Br(), html.Br(), html.Br(),
                            html.Br(), html.Br(), html.Br(),
                            html.Br(), html.Br(), html.Br(),
                            html.Br(), html.Br(), html.Br(),
                            html.Br(), html.Br(), html.Br(),
                            html.Br(), html.Br(), html.Br(),
                            html.Br(), html.Br(), html.Br(),
                            html.Br(), html.Br(), html.Br(),
                            
        ])

    return section




def section_2():
    """
    Création d'une section
    Les differentes classe CSS :
        - "titre_section" à affeter au titre de la section

    Pensez à modifier l'id du titre de la section pour qu'il coresponde au paramètre href du menu
    """
    section = html.Section(
                    className="section_content",
                    children=[
                        html.H3('Detection course output', id="partie2", className="titre_menu"),
                        html.P("The anomaly detection phase consists in two main steps:"),
                        html.Li(
                            "Step 1: based on the thresholds determined in the previous phase, “risky” cells are identified in each pre-cluster. Risky cells are cells with potential interference issues,"),
                        html.Li(
                            "Step 2: for each risky cell, we perform anomaly detection on the data to check if the cell actually presents interference issues."),

                        html.P("After detection course process, here is the table showing the detected anomalous cells with a comparison of the KPIs between anomalous and non-anomalous ranges."),

                        #conteneur.conteneur_2_graph(["main_output_table", "detailled_output_table"]),
                        html.Div(id='detailled_output_table'),
                        html.Br(),
                        html.Div(id='export_excel_button',style={'width': '14%', 'display': 'inline-block'}),
                        html.Div(id='export_excel_text', style={'width': '85%', 'display': 'inline-block'}),


                            

                    ])
    return section




def section_3():
    """
    Création d'une section
    Les differentes classe CSS :
        - "titre_section" à affeter au titre de la section

    Pensez à modifier l'id du titre de la section pour qu'il coresponde au paramètre href du menu
    """
    section = html.Section(
        className="section_content",
        children=[
            html.H3('Cells visualization', id="partie3", className="titre_menu"),
            html.P("After interference detection phase, interference cases are split into two categories:"),
            html.Li("PIM issues correspond to cases where a cell is detected as having interference issues while none of its neighbors has any issue,"),
            html.Li("We suppose that the interference cause is external if two or more neighboring cells are identified as interference victims."),
            html.P("You can filter on PIM or external interferences to visualize cells time series."),
            dcc.RadioItems(id='radio_pim',
                         options=[{'label': 'PIM', 'value': 'PIM'},
                                  {'label': 'External issues', 'value': 'External'}],
                         labelStyle={'display': 'inline-block'},
                         value='External'),
            html.Br(),
            html.Button(id='graph_button_3', children='Show elements', n_clicks=0),
            html.Br(), html.Br(),
            html.Li("the anomaly you want to be drawn,"),

              #html.Button(id='filter_button', children='Show cells', n_clicks=0),
              html.Div(id = 'dropdown_cell'),
              #html.Div(id = 'dropdown_sort'),

            html.Button(id='graph_button_pim', children='Show anomalies', n_clicks=0),

            html.Br(), html.Br(),
            #html.Div(id='riskcells_graph'),
            conteneur.conteneur_2_graph(["interf_table", "interf_graph"]),
            html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(),
        ])
    return section


def section_4():
    """
    Création d'une section
    Les differentes classe CSS :
        - "titre_section" à affeter au titre de la section

    Pensez à modifier l'id du titre de la section pour qu'il coresponde au paramètre href du menu
    """
    section = html.Section(
        className="section_content",
        children=[
            html.H3('KPI visualization', id="partie4", className="titre_menu"),
            html.P("Finally, click on 'Show graphs' button to visualize the behavior of the main KPIs during anomalous ranges."),
            html.Button(id='graph_button_4', children='Show graphs', n_clicks=0),
            html.Div(id='kpi_hsupa'),
            html.Div(id='kpi_hsdpa'),
            html.Div(id='kpi_3GTRAFFICSPEECH'),
            html.Div(id='kpi_3G&3G+ULTraffic'),
            html.Div(id='kpi_TotalDataTrafficDl&UL(GBytes)'),
            html.Div(id='kpi_dcrps'),
            html.Div(id='kpi_cssrcs'),
            html.Div(id='kpi_cssrps'),
                              html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br(),
                                html.Br(), html.Br(), html.Br()
                                
        ])
    return section
