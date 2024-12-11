# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:12:32 2023

@author: ythiriet
"""


# Function to create HTML page to ask user data to make prediction
def preparation():


    # Global importation
    from yattag import Doc
    from yattag import indent
    import joblib
    import numpy as np

    # Data importation
    ARRAY_DATA_ENCODE_REPLACEMENT = joblib.load("./script/data_replacement/array_data_encode_replacement.joblib")
    NAME_DATA_ENCODE_REPLACEMENT = np.zeros([ARRAY_DATA_ENCODE_REPLACEMENT.shape[0]], dtype = object)
    for i, ARRAY in enumerate(ARRAY_DATA_ENCODE_REPLACEMENT):
        NAME_DATA_ENCODE_REPLACEMENT[i] = ARRAY[0,0]

    # Setting list for prediction

    COLORS_RGB = [[102,51,0],[255,153,51],[0,0,0],[1,235,1],
                  [255,255,255],[0,0,120],[1,215,88],[255,255,51],
                  [255,0,0],[255,102,255],[148,129,43],[230,230,250],
                  [255,0,255],[110,75,38],[75,0,130],[255,215,0],
                  [250,128,114],[255,191,0],[64,224,208],[0,0,255],
                  [153,0,153],[160,160,160],[102,51,0],[0,0,0]]
    
    CAP_SHAPE = ARRAY_DATA_ENCODE_REPLACEMENT[np.where(NAME_DATA_ENCODE_REPLACEMENT == "cap-shape")[0][0]][:,-1]
    COLORS = ARRAY_DATA_ENCODE_REPLACEMENT[np.where(NAME_DATA_ENCODE_REPLACEMENT == "cap-color")[0][0]][:,-1]
    HABITATS = ARRAY_DATA_ENCODE_REPLACEMENT[np.where(NAME_DATA_ENCODE_REPLACEMENT == "habitat")[0][0]][:,-1]
    RING_TYPE = ARRAY_DATA_ENCODE_REPLACEMENT[np.where(NAME_DATA_ENCODE_REPLACEMENT == "ring-type")[0][0]][:,-1]
    SEASONS = ARRAY_DATA_ENCODE_REPLACEMENT[np.where(NAME_DATA_ENCODE_REPLACEMENT == "season")[0][0]][:,-1]
    


    # Creating HTML
    doc, tag, text, line = Doc().ttl()

    # Adding pre-head
    doc.asis('<!DOCTYPE html>')
    doc.asis('<html lang="fr">')
    with tag('head'):
        doc.asis('<meta charset="UTF-8">')
        doc.asis('<meta http-equiv="X-UA-Compatible" content = "IE=edge">')
        doc.asis('<link rel="stylesheet" href="./static/style.css">')
        doc.asis('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">')
        doc.asis('<meta name = "viewport" content="width=device-width, initial-scale = 1.0">')

    # Body start
    with tag('body', klass = 'background'):
        with tag('div', klass = "container"):
            with tag('div', klass = "row"):
                with tag('div', klass = "col-md-9"):
                    line('h1', 'Poisonous Mushroom prediction', klass = "text-center title")
                with tag('div', klass = "col"):
                    doc.asis('<img src="/static/classic_mushroom.jpg" alt="Mushroom" width=100% height=100% title="Mushroom"/>')
                
            # Launching prediction
            with tag('form', action = "{{url_for('treatment')}}", method = "POST", enctype = "multipart/form-data"):
                
                # Cap diameter
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"): 
                            with tag('div', klass = "col-md-6"):
                                line('p', 'Diametre du chapeau [cm]', klass = "p2")
                            with tag('div', klass = "col", style="text-align:center"):
                                doc.input(name = 'cap-diameter', type = 'text', size = "8", placeholder = "0",
                                          minlength = 1, klass = 'area_input')

                # Cap shape
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', 'Forme du chapeau', klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for SHAPE in CAP_SHAPE:
                                        with tag('div', klass = "col-md-4"):
                                            doc.input(name = 'cap-shape', type = 'radio', value = SHAPE, klass = 'radio_text')
                                            text(SHAPE)
                    
                    with tag('div', klass = "col"):
                        doc.asis('<img src="/static/cap_shape_mushroom.jpg" alt="Cap_shape" width=100% height=100% title="Cap_shape"/>')
                        
                
                # Cap color
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', 'Couleur majoritaire du chapeau', klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for i, COLOR in enumerate(COLORS):
                                        with tag('div', klass = "col-md-3", style = f"color:rgb({COLORS_RGB[i][0]},{COLORS_RGB[i][1]},{COLORS_RGB[i][2]}"):
                                            doc.input(name = 'cap-color', type = 'radio', value = COLOR, klass = 'radio_text')
                                            text(COLOR)

                    # Separation line
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "line_2"):
                            text('')
                
                
                # Does bruise or bleed (Hematome ou saignement)
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', 'Consistance du champignon', klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    with tag('div', klass = "col-md"):
                                        doc.input(name = 'does-bruise-or-bleed', type = 'radio', value = 1, klass = 'radio_text')
                                        text("Visqueux")
                                    with tag('div', klass = "col-md"):
                                        doc.input(name = 'does-bruise-or-bleed', type = 'radio', value = 0, klass = 'radio_text')
                                        text("Fluide")
                    
                    # Separation line
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "line_2"):
                            text('')
                
                
                # Gill color (lames sous le chapeau)
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', 'Couleur sous le chapeau', klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for i, COLOR in enumerate(COLORS):
                                        with tag('div', klass = "col-md-3", style = f"color:rgb({COLORS_RGB[i][0]},{COLORS_RGB[i][1]},{COLORS_RGB[i][2]}"):
                                            doc.input(name = 'gill-color', type = 'radio', value = COLOR, klass = 'radio_text')
                                            text(COLOR)
                    
                    with tag('div', klass = "col"):
                        doc.asis('<img src="/static/gill_mushroom.jpg" alt="gill" width=100% height=100% title="gill_mushroom"/>')
                    
                    # Separation line
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "line_2"):
                            text('')
                
                
                # Stem height
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"): 
                            with tag('div', klass = "col-md-6"):
                                line('p', 'Hauteur de la tige [cm]', klass = "p2")
                            with tag('div', klass = "col", style="text-align:center"):
                                doc.input(name = 'stem-height', type = 'text', size = "8", placeholder = "0",
                                          minlength = 1, klass = 'area_input')
                
                # Stem width
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"): 
                            with tag('div', klass = "col-md-6"):
                                line('p', 'Largeur de la tige [mm]', klass = "p2")
                            with tag('div', klass = "col", style="text-align:center"):
                                doc.input(name = 'stem-width', type = 'text', size = "8", placeholder = "0",
                                          minlength = 1, klass = 'area_input')
                
                # Stem color
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', 'Couleur de la tige', klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for i, COLOR in enumerate(COLORS):
                                        with tag('div', klass = "col-md-3", style = f"color:rgb({COLORS_RGB[i][0]},{COLORS_RGB[i][1]},{COLORS_RGB[i][2]}"):
                                            doc.input(name = 'stem-color', type = 'radio', value = COLOR, klass = 'radio_text')
                                            text(COLOR)
                    
                    # Separation line
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "line_2"):
                            text('')
                
                # Has ring
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', "Presence d'anneaux ?", klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    with tag('div', klass = "col-md"):
                                        doc.input(name = 'has-ring', type = 'radio', value = 1, klass = 'radio_text')
                                        text("Oui")
                                    with tag('div', klass = "col-md"):
                                        doc.input(name = 'has-ring', type = 'radio', value = 0, klass = 'radio_text')
                                        text("Non")
                
                # Ring type
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', 'Forme des anneaux', klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for TYPE in RING_TYPE:
                                        with tag('div', klass = "col-md-4"):
                                            doc.input(name = 'ring-type', type = 'radio', value = TYPE, klass = 'radio_text')
                                            text(TYPE)
                    
                    # Separation line
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "line_2"):
                            text('')
                
                # Habitat
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', 'Localisation', klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for HABITAT in HABITATS:
                                        with tag('div', klass = "col-md-4"):
                                            doc.input(name = 'habitat', type = 'radio', value = HABITAT, klass = 'radio_text')
                                            text(HABITAT)
                                            
                # Season
                with tag('div', klass = "row"):
                    with tag('div', klass = "col-md-9"):
                        with tag('div', klass = "row justify-content-around margin_1"):
                            with tag('div', klass = "col-md-4"):
                                line('p', 'Saison de developpement', klass = "p2")
                            with tag('div', klass = "col radio_text", style="text-align:center"):
                                with tag('div', klass = "row"):
                                    for SEASON in SEASONS:
                                        with tag('div', klass = "col-md-5"):
                                            doc.input(name = 'season', type = 'radio', value = SEASON, klass = 'radio_text')
                                            text(SEASON)

                

                # Submit button
                with tag('div', klass = "row"):
                    with tag('div', klass = "text-center div2"):
                        with tag('button', id = 'submit_button', name = "action", klass="btn btn-primary", value = 'Predict'):
                            text('Predict')
                                

    # Saving HTML created
    with open(f"./templates/predict.html", "w") as f:
        f.write(indent(doc.getvalue(), indentation = '    ', newline = '\n', indent_text = True))
        f.close()


# Function to make prediction and plotting them for the customer
def prediction(CURRENT_DIRECTORY, MODEL_INPUT_HTML, DATA_NAMES_HTML):

    # Global importation
    import joblib
    import numpy as np
    from yattag import Doc
    from yattag import indent
    import math
    import random
    
    # Global init
    RF_MODEL = False
    NN_MODEL = False
    GB_MODEL = False
    XG_MODEL = True
    REGRESSION = False

    # Class creation
    class Data_prediction():
        def __init__(self, MODEL):
            self.ARRAY_DATA_ENCODE_REPLACEMENT = joblib.load("./script/data_replacement/array_data_encode_replacement.joblib")
            self.DATA_NAMES = joblib.load("./script/data_replacement/data_names.joblib")
            
            self.MODEL = MODEL
            
            self.JS_CANVAS = ""
            self.JS_ANIMATION = ""

        
        def entry_data_arrangement(self, MODEL_INPUT_HTML, DATA_NAMES_HTML):
            self.MODEL_INPUT = np.zeros([self.DATA_NAMES.shape[0]], dtype = object)
            DATA_NAMES_HTML = np.array(DATA_NAMES_HTML)
            
            for i, DATA_NAME in enumerate(self.DATA_NAMES):
                self.MODEL_INPUT[i] = MODEL_INPUT_HTML[np.where(DATA_NAME == DATA_NAMES_HTML)[0][0]]


        # Turning word into numbers to make predictions
        def entry_data_modification(self):

            for i in range(self.MODEL_INPUT.shape[0]):
                for ARRAY in self.ARRAY_DATA_ENCODE_REPLACEMENT:
                    if self.DATA_NAMES[i] == ARRAY[0,0]:
                        for j in range(ARRAY.shape[0]):
                            if self.MODEL_INPUT[i] == ARRAY[j,3]:
                                self.MODEL_INPUT[i] = ARRAY[j,2]


        # Making prediction using model chosen
        def making_prediction(self, REGRESSION):
            self.PREDICTION = self.MODEL.predict(self.MODEL_INPUT.reshape(1,-1))
            print(self.PREDICTION)
            
            if REGRESSION == False:
                self.PROBA = self.MODEL.predict_proba(self.MODEL_INPUT.reshape(1,-1))
                print(self.PROBA)


        # Creating javascript using prediction
        def javascript_result_creation(self):
            
            mushroom_cap_width = 300
            
            # Creating Canvas to plot graphic
            self.JS_CANVAS += '/* Creation du canvas */\n'
            self.JS_CANVAS += 'var canvas = document.getElementById("canvas1");\n'
            self.JS_CANVAS += 'const width = (canvas.width = window.innerWidth);\n'
            self.JS_CANVAS += 'const height = (canvas.height = 500);\n'
            self.JS_CANVAS += f'const x = {int(self.PREDICTION[0]/1000)};\n'
            self.JS_CANVAS += 'const mushroom_center = width/4;\n'
            self.JS_CANVAS += f'const mushroom_cap_width = {mushroom_cap_width};\n'
            self.JS_CANVAS += 'const mushroom_cap_height_pos = 100;\n'
            self.JS_CANVAS += 'const mushroom_cap_small_elipse = 40;\n'
            self.JS_CANVAS += 'const mushroom_stem_height = 400;\n'
            self.JS_CANVAS += 'const mushroom_stem_width_max = 25;\n'
            self.JS_CANVAS += 'const mushroom_stem_width_min = 18;\n'
            self.JS_CANVAS += "canvas.style.position = 'relative';\n"
            self.JS_CANVAS += "canvas.style.zIndex = 1;"
            for i in range(12):
                self.JS_CANVAS += f'var ctx{i} = canvas.getContext("2d");\n'
            
            # Function to display correct format for number
            self.JS_CANVAS += '\n/* Fonction pour modifier le style des nombres affichés */\n'
            self.JS_CANVAS += 'function numberWithCommas(x) {\n'
            self.JS_CANVAS += '   return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");\n'
            self.JS_CANVAS += '}\n'
            
            # Function to display correct format for number
            self.JS_CANVAS += '\n/* Fonction pour generer un entier aleatoire entre 2 valeurs */\n'
            self.JS_CANVAS += 'function getRandomInt(min, max) {\n'
            self.JS_CANVAS += '    const minCeiled = Math.ceil(min);\n'
            self.JS_CANVAS += '    const maxFloored = Math.floor(max);\n'
            self.JS_CANVAS += '    return Math.floor(Math.random() * (maxFloored - minCeiled) + minCeiled);\n'
            self.JS_CANVAS += '}\n'
            
            # Function to change color
            self.JS_CANVAS += "\n/* Creation d'une fonction pour changer la couleur */\n"
            self.JS_CANVAS += 'function rgb(r, g, b){\n'
            self.JS_CANVAS += 'return "rgb("+r+","+g+","+b+")";\n'
            self.JS_CANVAS += '}\n'
            
            # Mushroom construction
            self.JS_CANVAS += "\n/* Creation de la base du champignon */\n"
            self.JS_CANVAS += f'ctx2.fillStyle = rgb({255*self.PROBA[0][1]*self.PROBA[0][1]*self.PROBA[0][1]},{255*self.PROBA[0][0]*self.PROBA[0][0]*self.PROBA[0][0]},0);\n'
            self.JS_CANVAS += 'ctx2.strokeStyle = "rgb(0,0,0)";\n'
            self.JS_CANVAS += 'ctx2.beginPath();\n'
            self.JS_CANVAS += 'ctx2.moveTo(mushroom_center - mushroom_cap_width/2, mushroom_cap_height_pos);\n'
            self.JS_CANVAS += 'ctx2.lineTo(mushroom_center + mushroom_cap_width/2, mushroom_cap_height_pos);\n'
            self.JS_CANVAS += 'ctx2.lineTo(mushroom_center - mushroom_cap_width/2, mushroom_cap_height_pos);\n'
            self.JS_CANVAS += 'ctx2.lineWidth = 1;\n'
            self.JS_CANVAS += 'ctx2.stroke();\n'
            
            # Mushroom construction
            self.JS_CANVAS += "\n/* Creation du chapeau du champignon */\n"
            self.JS_CANVAS += 'ctx2.strokeStyle = "rgb(0,0,0)";\n'
            self.JS_CANVAS += 'ctx2.ellipse(mushroom_center, mushroom_cap_height_pos, mushroom_cap_width/2, mushroom_cap_small_elipse, Math.PI, 0, Math.PI);\n'
            self.JS_CANVAS += 'ctx2.stroke();\n'
            self.JS_CANVAS += "\n/* Creation des cotes du pied */\n"
            self.JS_CANVAS += 'ctx2.strokeStyle = "rgb(0,0,0)";\n'
            self.JS_CANVAS += 'ctx2.moveTo(mushroom_center - mushroom_stem_width_max, mushroom_stem_height);\n'
            self.JS_CANVAS += 'ctx2.bezierCurveTo(mushroom_center - ((5/4)*mushroom_stem_width_max - mushroom_stem_width_min/4), mushroom_cap_height_pos + (8/10)*(mushroom_stem_height - mushroom_cap_height_pos), mushroom_center - ((11/8)*mushroom_stem_width_max - mushroom_stem_width_min/2), mushroom_cap_height_pos + (6/10)*(mushroom_stem_height - mushroom_cap_height_pos), mushroom_center - mushroom_stem_width_min, mushroom_cap_height_pos + (3/10)*(mushroom_stem_height - mushroom_cap_height_pos));\n'
            self.JS_CANVAS += 'ctx2.lineTo(mushroom_center - mushroom_stem_width_min, mushroom_cap_height_pos);\n'
            self.JS_CANVAS += 'ctx2.lineTo(mushroom_center + mushroom_stem_width_min, mushroom_cap_height_pos);\n'
            self.JS_CANVAS += 'ctx2.lineTo(mushroom_center + mushroom_stem_width_min, mushroom_cap_height_pos + (3/10)*(mushroom_stem_height - mushroom_cap_height_pos));\n'
            self.JS_CANVAS += 'ctx2.bezierCurveTo(mushroom_center + ((11/8)*mushroom_stem_width_max - mushroom_stem_width_min/2), mushroom_cap_height_pos + (6/10)*(mushroom_stem_height - mushroom_cap_height_pos), mushroom_center + ((5/4)*mushroom_stem_width_max - mushroom_stem_width_min/4), mushroom_cap_height_pos + (8/10)*(mushroom_stem_height - mushroom_cap_height_pos), mushroom_center + mushroom_stem_width_max, mushroom_stem_height);\n'
            self.JS_CANVAS += 'ctx2.bezierCurveTo(mushroom_center + (4/5)*mushroom_stem_width_max, mushroom_stem_height*1.03, mushroom_center + mushroom_stem_width_min, mushroom_stem_height*1.035, mushroom_center, mushroom_stem_height*1.04);\n'
            self.JS_CANVAS += 'ctx2.bezierCurveTo(mushroom_center - mushroom_stem_width_min, mushroom_stem_height*1.035, mushroom_center - (4/5)*mushroom_stem_width_max, mushroom_stem_height*1.03, mushroom_center - mushroom_stem_width_max, mushroom_stem_height);\n'
            self.JS_CANVAS += 'ctx2.stroke();\n'
            self.JS_CANVAS += 'ctx2.fill();\n'
            self.JS_CANVAS += "\n/* Creation de petites taches */\n"
            self.JS_CANVAS += 'ctx3.fillStyle = "rgb(255,255,255)";\n'
            
            for i in range(int(mushroom_cap_width/14)):
                self.JS_CANVAS += 'ctx3.beginPath();\n'
                self.JS_CANVAS += f'ctx3.ellipse(mushroom_center - mushroom_cap_width/2.2 + {2*i/int(mushroom_cap_width/14)*mushroom_cap_width/2.2 + random.uniform(-1,1)}, 0.98*mushroom_cap_height_pos + {random.uniform(-3,-1)}, 3, 2, 2*Math.PI, 0, 2*Math.PI);\n'
                self.JS_CANVAS += 'ctx3.stroke();\n'
                self.JS_CANVAS += 'ctx3.fill();\n'
                self.JS_CANVAS += 'ctx3.beginPath();\n'
                self.JS_CANVAS += f'ctx3.ellipse(mushroom_center - mushroom_cap_width/2.5 + {2*i/int(mushroom_cap_width/14)*mushroom_cap_width/2.5 + random.uniform(-1,1)}, 0.90*mushroom_cap_height_pos + {random.uniform(-3,-1)}, 3, 2, 2*Math.PI, 0, 2*Math.PI);\n'
                self.JS_CANVAS += 'ctx3.stroke();\n'
                self.JS_CANVAS += 'ctx3.fill();\n'
            for i in range(int(mushroom_cap_width/16)):
                self.JS_CANVAS += 'ctx3.beginPath();\n'
                self.JS_CANVAS += f'ctx3.ellipse(mushroom_center - mushroom_cap_width/2.8 + {2*i/int(mushroom_cap_width/16)*mushroom_cap_width/2.8 + random.uniform(-1,1)}, 0.82*mushroom_cap_height_pos + {random.uniform(-3,-1)}, 3, 2, 2*Math.PI, 0, 2*Math.PI);\n'
                self.JS_CANVAS += 'ctx3.stroke();\n'
                self.JS_CANVAS += 'ctx3.fill();\n'
            for i in range(int(mushroom_cap_width/22)):
                self.JS_CANVAS += 'ctx3.beginPath();\n'
                self.JS_CANVAS += f'ctx3.ellipse(mushroom_center - mushroom_cap_width/3.5 + {2*i/int(mushroom_cap_width/22)*mushroom_cap_width/3.5 + random.uniform(-1,1)}, 0.72*mushroom_cap_height_pos + {random.uniform(-3,-1)}, 3, 2, 2*Math.PI, 0, 2*Math.PI);\n'
                self.JS_CANVAS += 'ctx3.stroke();\n'
                self.JS_CANVAS += 'ctx3.fill();\n'
                

            # Percentage text animation
            self.JS_CANVAS += "\n/* Creation d'une animation pour afficher la probabilite progressivement */\n"
            self.JS_CANVAS += "ctx4.fillStyle = 'rgb(0,0,50)';\n"
            self.JS_CANVAS += 'ctx4.font = "48px georgia";\n'
            self.JS_CANVAS += f'var proba_poisonous = {self.PROBA[0][1]};\n'
            self.JS_CANVAS += f'var textString = "{int(100*self.PROBA[0][1])} %",\n'
            self.JS_CANVAS += "    textWidth = ctx3.measureText(textString).width;\n"
            self.JS_CANVAS += 'var id2 = null;\n'
            self.JS_CANVAS += 'function myPercentage() {\n'
            self.JS_CANVAS += '  var percentage = 0;\n'
            self.JS_CANVAS += f'   const x2 = {int(100*self.PROBA[0][1])};\n'
            self.JS_CANVAS += '  id4 = setInterval(frame2, 50);\n'
            self.JS_CANVAS += '  function frame2() {\n'
            self.JS_CANVAS += '    if (percentage < x2) { \n'
            self.JS_CANVAS += '      ctx4.clearRect(2*width/5,200,3*width/4,100);\n'
            self.JS_CANVAS += '      percentage += getRandomInt(0.01, 1);\n'
            self.JS_CANVAS += '      var textString2 = `${percentage} %`;\n'
            self.JS_CANVAS += "          textWidth2 = ctx3.measureText(textString2).width;\n"
            self.JS_CANVAS += '      ctx4.fillText(numberWithCommas(textString2),width/2,260);\n'
            self.JS_CANVAS += '    } else { \n'
            self.JS_CANVAS += '      ctx4.fillStyle = rgb(255*proba_poisonous*proba_poisonous*proba_poisonous,255*(1-proba_poisonous)*(1-proba_poisonous)*(1-proba_poisonous),0);\n'
            self.JS_CANVAS += '      ctx4.clearRect(2*width/5,200,3*width/4,100);\n'
            self.JS_CANVAS += '      ctx4.fillText("Poisonous probability",15*width/40,210);\n'
            self.JS_CANVAS += '      ctx4.fillText(numberWithCommas(textString),width/2,260);\n'
            self.JS_CANVAS += '      if (proba_poisonous > 0.5) { \n'
            self.JS_CANVAS += '          ctx4.fillText("Poisonous",mushroom_center - 110,50);\n'
            self.JS_CANVAS += '      } else { \n'
            self.JS_CANVAS += '          ctx4.fillText("Safe",mushroom_center - 45,50);\n'
            self.JS_CANVAS += '      }\n'
            self.JS_CANVAS += '      clearInterval(id);\n'
            self.JS_CANVAS += '    }\n'
            self.JS_CANVAS += '  }\n'
            self.JS_CANVAS += '}\n'
            
            # Writing Javascript into a file
            with open("./static/canevas.js","w") as f:
                f.write(self.JS_CANVAS)
    

        # Creating html
        def html_result_creation(self, CURRENT_DIRECTORY):

            # Creating HTML
            doc, tag, text, line = Doc(defaults = {'Month': 'Fevrier'}).ttl()

            doc.asis('<!DOCTYPE html>')
            doc.asis('<html lang="fr">')
            with tag('head'):
                doc.asis('<meta charset="UTF-8">')
                doc.asis('<meta http-equiv="X-UA-Compatible" content = "IE=edge">')
                doc.asis('<link rel="stylesheet" href="./static/style.css">')
                doc.asis('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">')
                doc.asis('<meta name = "viewport" content="width=device-width, initial-scale = 1.0">')

            # Body start
            with tag('body', klass = 'background'):
                with tag('div', klass = "container"):
                    with tag('div', klass = "row"):
                        with tag('div', klass = "col-md-9"):
                            line('h1', 'Poisonous Mushroom prediction', klass = "text-center title")
                        with tag('div', klass = "col"):
                            doc.asis('<img src="/static/classic_mushroom.jpg" alt="Mushroom" width=100% height=100% title="Mushroom"/>')


                    with tag('div', klass="col"):
                        with tag('canvas', id = "canvas1", width="540", height="600"):
                            text("")
                    
                            # Script for canvas
                            doc.asis('<script src="/static/canevas.js"></script>')
                
                        # Launching script when arriving on the page
                        with tag('script', type="text/javascript"):
                            text('myPercentage();')
                
                # Button to go back to previous page
                with tag('form', action = "{{url_for('predict')}}", method = "GET", enctype = "multipart/form-data"):
                    with tag('div', klass = "text-center"):
                        with tag('button', id = 'submit_button', name = "action", klass="btn btn-primary", value = 'Go back to previous page'):
                            line('p1', '')
                            text('Go back to previous page')

            # Saving HTML
            with open("./templates/result.html", "w") as f:
                f.write(indent(doc.getvalue(), indentation = '    ', newline = '\n', indent_text = True))
                f.close()

    # Loading models
    if RF_MODEL == True:
        with open("./script/models/rf_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif NN_MODEL == True:
        with open("./script/models/nn_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif GB_MODEL == True:
        with open("./script/models/gb_model.sav", 'rb') as f:
            MODEL = joblib.load(f)
    elif XG_MODEL == True:
        with open("./script/models/xg_model.sav", 'rb') as f:
            MODEL = joblib.load(f)

    # Personnalized prediction
    global_data_prediction = Data_prediction(MODEL)
    global_data_prediction.entry_data_arrangement(MODEL_INPUT_HTML, DATA_NAMES_HTML)
    global_data_prediction.entry_data_modification()

    global_data_prediction.making_prediction(REGRESSION)
    global_data_prediction.javascript_result_creation()
    global_data_prediction.html_result_creation(CURRENT_DIRECTORY)
