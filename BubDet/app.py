import sys
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from model import Detectron_Model, plot_prediction
sys.path.insert(0, '../')
from utils import *
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import base64
from io import BytesIO

# Init detectron2 model
#MODEL_PATH = 'model_final.pth'
#model = Detectron_Model(model_path = MODEL_PATH)

def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    return dbc.Row([dbc.Col(title, md=8)])

def image_card(src, header=None):
    return dbc.Card(
        [
            dbc.CardHeader(header),
            dbc.CardBody(html.Img(src=src, style={"width": "100%"})),
        ]
    )

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

models_dropdown = dcc.Dropdown(
    id="models-dropdown",
    options=[
        {"label": "RetinaNet", "value": "retinanet"},
        {"label": "FasterRCNN", "value": "faster_rcnn"},
    ],
    value="retinanet"  # Domyślny wybór modelu
)

controls = [
    dcc.Upload(
        dbc.Card(
            "Drag and Drop or Click",
            body=True,
            style={
                "textAlign": "center",
                "borderStyle": "dashed",
                "borderColor": "black",
            },
        ),
        id="img-upload",
        multiple=False,
    ),
     models_dropdown 
    
]

app.layout = dbc.Container(
    [
        Header("Bubbles detection", app),
        html.Hr(),
        dbc.Row([dbc.Col(c) for c in controls]),
        html.Br(),
        dbc.Spinner(
            # this is list for image components and button below
            # can be replaced by traditional list as [Div(img), Div(img), Button()]
            dbc.Row([
                    *[dbc.Col(html.Div(id=img_id)) for img_id in ["original-img","prediction-image"]],
                    html.Button(id='submit-button-state', n_clicks=0, children='Predict')
                ]
            )
        )
    ],
    fluid=False,
)


@app.callback(
    [Output("original-img", "children")],
    [Input("img-upload", "contents")],
    [State("img-upload", "filename")],
)
def upload_image(img_str, filename):
    if img_str is None:
        return dash.no_update
    
    lr = image_card(img_str, header="Original Image")

    # Decode the image
    #image = decode_image(img_str)

    # Get image bytes from base64 string
    #img_bytes = base64.b64decode(img_str.split("base64,")[-1])

    # Extract EXIF data
    #exif_data = get_exif_data(img_bytes)

    # Print EXIF data
    #print("Aperture:", exif_data.get("EXIF ApertureValue"))
    #print("Exposure Bias:", exif_data.get("EXIF ExposureBiasValue"))
    #print("Focal Length:", exif_data.get("EXIF FocalLength"))


    return lr,


# It seems that uploaded image is "stated" so can be used in other method
# Method needs trigger such as button, it needs to be signes as "Input"
@app.callback([Output('prediction-image', 'children'),
               Input('submit-button-state', 'n_clicks'),
               State("img-upload", "contents"), State("img-upload", "filename"),State('models-dropdown', 'value')]
            )
def predict(n_clicks, img_str, filename,selected_model):
    if img_str is None:
        return dash.no_update
    
    if selected_model == 'retinanet':
        model = Detectron_Model(model_path='retina.pth',model_type='retinanet')
    elif selected_model == 'faster_rcnn':
        model = Detectron_Model(model_path='rcnn.pth',model_type='faster_rcnn')

    image = decode_image(img_str = img_str)
    pred = model.predict(image)
    FocalLength = get_meta_info(img_str = img_str)
    image_with_boxes = plot_prediction(image, pred, FocalLength)
    b64_string = encode_image(image_with_boxes)
    sr = image_card(b64_string, header="Prediction Image")
    return sr,

if __name__ == '__main__':
    app.run_server(debug=True)