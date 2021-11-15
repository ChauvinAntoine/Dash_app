import dash_html_components as html
import dash_core_components as dcc


def conteneur_3_infobox(images_url, titres, id_output):
    """Créer un groupe de 3 infobox sur une ligne

    Parameters
    ----------
    images_url : list (liste de 3 images)
    titres : list (Liste des 3 titres)
    id_output : list (liste des 3 output id
    """
    conteneur = html.Div(
        className="groupe_3_infobox",
        children=[
            ########## Infobox 1
            html.Div(
                className="infobox_div",
                children=[
                    html.Div([
                        html.Img(className="image_infobox", src=images_url[0])
                    ], className="infobox_image_div"),
                    html.Div([
                        html.P(titres[0], className="text_infobox"),
                        html.P(id=id_output[0], children=["..."], className="text_infobox")
                    ], className="infobox_text_div")
                ]),
            ########## Infobox 2
            html.Div(
                className="infobox_div",
                children=[
                    html.Div([html.Img(className="image_infobox",src=images_url[1])
                        ], className="infobox_image_div"),
                    html.Div([
                        html.P(titres[1], className="text_infobox"),
                        html.P(id=id_output[1], children=["..."], className="text_infobox")
                    ], className="infobox_text_div")
                ]),
            ########## Infobox 3
            html.Div(
                className="infobox_div",
                children=[
                    html.Div([html.Img(className="image_infobox", src=images_url[2])
                        ], className="infobox_image_div"),
                    html.Div([
                        html.P(titres[2], className="text_infobox"),
                        html.P(id=id_output[2], children=["..."], className="text_infobox")
                    ], className="infobox_text_div")
                ])
        ])
    return conteneur


def conteneur_1_graph(graph_id):
    """Créer un block pour inserer un graphique

    Parameters
    ----------
    graph_id : str (id output du graphique)
    """
    conteneur = html.Div(
        className="conteneur_graph",
        children=[
            dcc.Graph(id=graph_id)
        ])
    return conteneur



def conteneur_2_graph(liste_graph_id):
    """Créer un block pour inserer deux graphiques l'un à coté de l'autre

    Parameters
    ----------
    graph_id : list (liste de deux id output du graphique)
    """
    conteneur = html.Div(
        className="conteneur_2_graph",
        children=[
            html.Div(className="conteneur_2G_graph", id=liste_graph_id[0]),
            html.Div(className="conteneur_2G_graph", id=liste_graph_id[1])
        ])
    return conteneur

