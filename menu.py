
import dash_html_components as html
import dash_core_components as dcc
import base64

# image_filename_logo = 'Orange.png'
# encoded_image = base64.b64encode(open(image_filename_logo, 'rb').read()).decode('ascii')

def lef_menu():
    """Création du menu gauche du dashboard
    Les differentes classe CSS :
        - "logo" à affecter aux logos ou images du menu
        - "titre1_menu" à affeter aux menu de type 1 du menu
        - "menu_lien_boite" à affceter aux balise LI des élements du menu
        - "menu_lien_text" à affecter aux balise A des éléments du menu
        - "titre_filtre_menu" à affceter aux titres des filtres

    Pour faire le lien vers les differentes sections du dashboard, pensez à modifier le paramètre href des balse A pour quelle renvoit vers l'id des titre de section
    """
    menu = html.Div(
            className="left_menu",
            children=[

                ###### Logo B&D
                html.Div([html.Img(className="logo", src="https://c.woopic.com/logo-orange.png", style={'height':'15%', 'width':'60%'})],
                         style={'textAlign':'center'}),
                #html.Img(className="logo", src='data:image/png;base64,{}'.format(encoded_image)),
                html.Br(),
                ###### Menu
                html.H1("Menu", className="titre1_menu"),
                # Lien vers les sections du dashbaord
                html.Ul(
                    className="menu_lien",
                    children=[
                        # Section 0
                        # html.Li([
                        #     html.A(className="menu_lien_text", href="#partie00", children="Cover page")
                        # ], className="menu_lien_boite"),
                        # Section 01
                        html.Li([
                            html.A(className="menu_lien_text", href="#partie01", children="Input data")
                        ], className="menu_lien_boite"),
                        # Section 0
                        html.Li([
                            html.A(className="menu_lien_text", href="#partie0", children="Detection course parameters")
                        ],className="menu_lien_boite"),
                        # Section 1
                        html.Li([
                            html.A(className="menu_lien_text", href="#partie1", children="Clustering")
                        ],className="menu_lien_boite"),
                        # Section 2
                        html.Li([
                            html.A(className="menu_lien_text", href="#partie2", children="Detection course output")
                        ],className="menu_lien_boite"),
                        # Section 3
                        html.Li([
                            html.A(className="menu_lien_text", href="#partie3", children="Cells visualization")
                        ],className="menu_lien_boite"),
                        # Section 4
                        # html.Li([
                        #     html.A(className="menu_lien_text", href="#partie6", children="Interference localization")
                        # ], className="menu_lien_boite"),
                        html.Li([
                            html.A(className="menu_lien_text", href="#partie4", children="KPI visualization")
                        ],className="menu_lien_boite"),
                    ]
                ),

    ])

    return menu