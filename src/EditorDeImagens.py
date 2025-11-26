# Trabalho do GB de Processamento Gráfico: Fundamentos 2025/2
# Integrantes: Cássio F. Braga, Gabriel C. Walber e Patrícia Nagel
# Prof: Rossana Baptista Queiroz

# Precisa instalar o OpenCV e Dear PyGui antes de rodar este código:
# pip install opencv-python dearpygui

# Importações
import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
import os
from enum import IntEnum

# Constantes globais
window_size_width = 1280
window_size_height = 720
effects_window_width = 400
effects_window_height = 600

# Variáveis globais
width_cam = 640
height_cam = 480
software_mode = "image"  # "image" ou "camera"
height, width, channels = 0, 0, 0
cv_image = None
image_data = None
mouse_pos = (0, 0)

# Listando stickers disponíveis
stickers_folder = "res/stickers"
stickers_files = [f for f in os.listdir(stickers_folder) if os.path.isfile(os.path.join(stickers_folder, f))]

# Array para armazenar os efeitos aplicados em ordem
effects_array = []

# Listas de nomes e tags dos efeitos
effects_list_names = ["Blur Gaussiano", "Escala de Cinza", "Detecção de Bordas", "Canal de Cor", "Sharpen", "Inverter Cores", "Brilho", "Contraste", "Saturação", "Laplaciano", "Sticker"]
effects_tags_names = ["blur_slider", "gray_scale_checkbox", "edge_detection_checkbox", "color_channel_selector", "sharpen_checkbox", "invert_colors_checkbox", "brightness_slider", "contrast_slider", "saturation_slider", "laplacian_checkbox", "sticker_selector"]
# Enum para os tipos de efeitos
class Effects(IntEnum):
    BLUR = 0
    GRAYSCALE = 1
    EDGE_DETECTION = 2
    CHANNEL_COLOR = 3
    SHARPEN = 4
    INVERT_COLORS = 5
    BRIGHTNESS = 6
    CONTRAST = 7
    SATURATION = 8
    LAPLACIAN = 9
    STICKER = 10

# Classe para representar um efeito aplicado
class Efeitos():
    def __init__(self, effect_type: Effects, tagname: str):
        self.effect_type = effect_type
        self.tagname = tagname

# Callback para remover um efeito da lista de efeitos da tela
def remove_effect_callback(sender, app_data, user_data):
    dpg.delete_item(effects_tags_names[user_data])
    dpg.delete_item("remove_" + effects_tags_names[user_data])
    dpg.delete_item("sep_" + effects_tags_names[user_data])
    effects_array.remove(next(e for e in effects_array if e.effect_type == user_data))
    update_texture()

# Função para adicionar um efeito com checkbox
def add_effect_checkbox(effect, effect_index):
    if any(e.effect_type == effect_index for e in effects_array):
            return  # Efeito já adicionado
    effects_array.append(Efeitos(effect_index, effects_tags_names[effect_index]))
    dpg.add_button(label="Remover " + effects_list_names[effect_index], tag="remove_" + effects_tags_names[effect_index], parent="effects_window", callback=remove_effect_callback, user_data=effect_index)
    dpg.add_checkbox(label=effect, tag=effects_tags_names[effect_index], callback=update_texture, parent="effects_window")
    dpg.add_separator(parent="effects_window", tag="sep_" + effects_tags_names[effect_index])

# Função para adicionar um efeito com slider
def add_effect_slider(effect, effect_index, min_value, max_value, default_value):
    if any(e.effect_type == effect_index for e in effects_array):
            return  # Efeito já adicionado
    effects_array.append(Efeitos(effect_index, effects_tags_names[effect_index]))
    dpg.add_button(label="Remover " + effects_list_names[effect_index], tag="remove_" + effects_tags_names[effect_index], parent="effects_window", callback=remove_effect_callback, user_data=effect_index)
    dpg.add_slider_int(label=effect, min_value=min_value, max_value=max_value, default_value=default_value, tag=effects_tags_names[effect_index], callback=update_texture, parent="effects_window")
    dpg.add_separator(parent="effects_window", tag="sep_" + effects_tags_names[effect_index])

# Função para adicionar um efeito com combo box
def add_effect_combo(effect, effect_index, items):
    if any(e.effect_type == effect_index for e in effects_array):
            return  # Efeito já adicionado
    effects_array.append(Efeitos(effect_index, effects_tags_names[effect_index]))
    dpg.add_button(label="Remover " + effects_list_names[effect_index], tag="remove_" + effects_tags_names[effect_index], parent="effects_window", callback=remove_effect_callback, user_data=effect_index)
    dpg.add_combo(label=effect, items=items, tag=effects_tags_names[effect_index], callback=update_texture, parent="effects_window")
    dpg.add_separator(parent="effects_window", tag="sep_" + effects_tags_names[effect_index])

# Callback para adicionar um efeito à lista de efeitos da tela
def add_effect_callback():
    effect = dpg.get_value("effect_selector")
    effect_index = effects_list_names.index(effect)

    if effect_index == Effects.GRAYSCALE:
        add_effect_checkbox(effect, effect_index)
        print("Adiciona filtro de escala de cinza. Converte a imagem para tons de cinza.")
        
    elif effect_index == Effects.BLUR:
        add_effect_slider(effect, effect_index, 0, 30, 0)
        print("Adiciona filtro de blur gaussiano. O valor do slider define o nível de desfoque aplicado à imagem.")
        
    elif effect_index == Effects.EDGE_DETECTION:
        add_effect_checkbox(effect, effect_index)
        print("Adiciona filtro de detecção de bordas. Imagem ficará em preto e branco destacando as bordas.")

    elif effect_index == Effects.CHANNEL_COLOR:
        add_effect_combo(effect, effect_index, ["R", "G", "B"])
        print("Adiciona filtro de canal de cor. R = Vermelho, G = Verde, B = Azul. Imagem ficará em tons de cinza com base no canal selecionado.")

    elif effect_index == Effects.SHARPEN:
        add_effect_checkbox(effect, effect_index)
        print("Adiciona filtro de sharpen. Aumenta a nitidez da imagem.")

    elif effect_index == Effects.INVERT_COLORS:
        add_effect_checkbox(effect, effect_index)
        print("Adiciona filtro de inverter cores. Inverte as cores da imagem.")

    elif effect_index == Effects.BRIGHTNESS:
        add_effect_slider(effect, effect_index, -255, 255, 0)
        print("Adiciona filtro de brilho. O valor do slider define o nível de brilho aplicado à imagem.")

    elif effect_index == Effects.CONTRAST:
        add_effect_slider(effect, effect_index, -100, 400, 0)
        print("Adiciona filtro de contraste. O valor do slider define o nível de contraste aplicado à imagem.")

    elif effect_index == Effects.SATURATION:
        add_effect_slider(effect, effect_index, 0, 255, 0)
        print("Adiciona filtro de saturação. O valor do slider define o nível de saturação aplicado à imagem.")

    elif effect_index == Effects.LAPLACIAN:
        add_effect_checkbox(effect, effect_index)
        print("Adiciona filtro de laplaciano. Destaca as áreas de rápida mudança de intensidade na imagem.")

    elif effect_index == Effects.STICKER:
        add_effect_combo(effect, effect_index, stickers_files)
        print("Adiciona sticker à imagem. Seleciona um sticker para sobrepor na imagem.")

# Função para processar os efeitos na imagem
def process_effects(image, effects_list):
    for effect in effects_list:
        value = dpg.get_value(effect.tagname)

        if effect.effect_type == Effects.GRAYSCALE:
            if value:
                image = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
                image = cv.cvtColor(image, cv.COLOR_GRAY2RGBA)
        elif effect.effect_type == Effects.BLUR:
            value = int(value)
            if value % 2 == 0:
                value += 1
            image = cv.GaussianBlur(image, (value, value), 0)
        elif effect.effect_type == Effects.EDGE_DETECTION:
            if value:
                gray = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
                edges = cv.Canny(gray, 100, 200)
                image = cv.cvtColor(edges, cv.COLOR_GRAY2RGBA)

        elif effect.effect_type == Effects.CHANNEL_COLOR:
            channel = value
            zeros = np.zeros(image.shape[:2], dtype="uint8")
            if channel == "B":
                image = cv.merge([zeros, zeros, image[:,:,2], image[:,:,3]])
            elif channel == "G":
                image = cv.merge([zeros, image[:,:,1], zeros, image[:,:,3]])
            elif channel == "R":
                image = cv.merge([image[:,:,0], zeros, zeros, image[:,:,3]])

        elif effect.effect_type == Effects.SHARPEN:
            if value:
                kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
                image = cv.filter2D(image, -1, kernel)

        elif effect.effect_type == Effects.INVERT_COLORS:
            if value:
                inverted = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
                inverted = cv.bitwise_not(inverted)
                image = cv.cvtColor(inverted, cv.COLOR_RGB2RGBA)

        elif effect.effect_type == Effects.BRIGHTNESS:
            brightness = int(value)
            if brightness != 0:
                hsv = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
                hsv = cv.cvtColor(hsv, cv.COLOR_RGB2HSV)
                h, s, v = cv.split(hsv)
                v = cv.add(v, brightness)
                v = np.clip(v, 0, 255)
                final_hsv = cv.merge((h, s, v))
                image = cv.cvtColor(final_hsv, cv.COLOR_HSV2RGB)
                image = cv.cvtColor(image, cv.COLOR_RGB2RGBA)

        elif effect.effect_type == Effects.CONTRAST:
            contrast = int(value)
            if contrast != 0:
                alpha_c = contrast/100 +1
                gamma_c = -contrast
                image = cv.addWeighted(image, alpha_c, image, 0, gamma_c)

        elif effect.effect_type == Effects.SATURATION:
            saturation = int(value)
            if saturation != 0:
                hsv = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
                hsv = cv.cvtColor(hsv, cv.COLOR_RGB2HSV)
                h, s, v = cv.split(hsv)
                s = cv.add(s, saturation)
                s = np.clip(s, 0, 255)
                final_hsv = cv.merge((h, s, v))
                image = cv.cvtColor(final_hsv, cv.COLOR_HSV2RGB)
                image = cv.cvtColor(image, cv.COLOR_RGB2RGBA)

        elif effect.effect_type == Effects.LAPLACIAN:
            if value:
                gray = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
                laplacian = cv.Laplacian(gray, cv.CV_64F)
                laplacian = cv.convertScaleAbs(laplacian)
                image = cv.cvtColor(laplacian, cv.COLOR_GRAY2RGBA)

        elif effect.effect_type == Effects.STICKER:
            sticker_file = value
            if sticker_file in stickers_files:
                sticker_path = os.path.join(stickers_folder, sticker_file)
                sticker_img = cv.imread(sticker_path, cv.IMREAD_UNCHANGED)
                if sticker_img is not None:
                    sticker_img = cv.cvtColor(sticker_img, cv.COLOR_BGRA2RGBA)
                    sticker_img = cv.resize(sticker_img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
                    sticker_height, sticker_width = sticker_img.shape[:2]
                    x, y = mouse_pos
                    x = int(x)
                    y = int(y)
                    x = x - sticker_width // 2
                    y = y - sticker_height // 2
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if x + sticker_width > image.shape[1]:
                        x = image.shape[1] - sticker_width
                    if y + sticker_height > image.shape[0]:
                        y = image.shape[0] - sticker_height
                    # Garantir que o sticker cabe na imagem
                    roi = image[y:y+sticker_height, x:x+sticker_width]
                    alpha_sticker = sticker_img[:, :, 3] / 255.0
                    alpha_bg = 1.0 - alpha_sticker
                    for c in range(3):  # Sobre os canais R, G, B
                        roi[:, :, c] = (alpha_sticker * sticker_img[:, :, c] + alpha_bg * roi[:, :, c]).astype(np.uint8)
                    roi[:, :, 3] = 255  # Definir alfa como opaco
                    image[y:y+sticker_height, x:x+sticker_width] = roi

    return image

# Callback para mudar o modo entre imagem e câmera
def change_mode_callback():
    global software_mode
    global width_cam
    global height_cam
    dpg_value = dpg.get_value("image_camera_selector")
    if dpg_value == "Câmera":
        software_mode = "camera"
        dpg.configure_item("opencv_video", width=width_cam, height=height_cam)
    else:
        software_mode = "image"
        height, width, channels = cv_image.shape
        dpg.configure_item("opencv_image", width=width, height=height)
        update_texture()

    dpg.configure_item("image_window", show=(software_mode == "image"))
    dpg.configure_item("video_window", show=(software_mode == "camera"))

    dpg.configure_item("image_camera_selector")

# Atualiza o frame de vídeo da câmera com os efeitos aplicados
def update_video_frame():
    if software_mode != "camera":
        return

    ret, frame = capture.read()
    if ret:
        # Converte o frame de BGR para RGBA
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        # Aplica os efeitos
        frame_rgb = process_effects(frame_rgb, effects_array)

        # Normaliza o frame para [0.0, 1.0]
        frame_normalized = frame_rgb.flatten() / 255.0

        # Atualiza a textura dinâmica
        dpg.set_value("opencv_video", frame_normalized)

# Atualiza a textura da imagem com os efeitos aplicados
def update_texture():
    if software_mode == "image":
        modified_image = cv_image.copy()
        modified_image = process_effects(modified_image, effects_array)
        dpg.set_value("opencv_image", modified_image.flatten() / 255.0)
        return

# Salva a imagem atual com os efeitos aplicados, seja da imagem ou do vídeo
def save_image_callback():
    if software_mode == "image":
        config = dpg.get_item_configuration("opencv_image")
        width = config["width"]
        height = config["height"]
        data_img = dpg.get_value("opencv_image")
        image_to_save = (np.array(data_img).reshape((height, width, 4)) * 255).astype(np.uint8)
        cv.imwrite("output_image.png", cv.cvtColor(image_to_save, cv.COLOR_RGBA2BGR))
    else:
        config = dpg.get_item_configuration("opencv_video")
        width = config["width"]
        height = config["height"]
        data_img = dpg.get_value("opencv_video")
        image_to_save = (np.array(data_img).reshape((height, width, 4)) * 255).astype(np.uint8)
        cv.imwrite("output_image.png", cv.cvtColor(image_to_save, cv.COLOR_RGBA2BGR))

# Callback para selecionar uma imagem do sistema de arquivos
def select_image_callback(sender, app_data):
    global image_path, cv_original_image, cv_image, height, width, channels, image_data
    image_path = app_data['file_path_name']
    cv_original_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)

    # Converte para RGBA
    if len(cv_original_image.shape) == 2:
        # Grayscale para RGBA
        cv_image = cv.cvtColor(cv_original_image, cv.COLOR_GRAY2RGBA)
    elif cv_original_image.shape[2] == 3:
        # BGR para RGBA
        cv_image = cv.cvtColor(cv_original_image, cv.COLOR_BGR2RGBA)
    elif cv_original_image.shape[2] == 4:
        # BGRA para RGBA
        cv_image = cv.cvtColor(cv_original_image, cv.COLOR_BGRA2RGBA)
    else:
        print("Formato de imagem não suportado.")
        return

    # Pega as dimensões da textura existente
    config = dpg.get_item_configuration("opencv_image")
    texture_width = config["width"]
    texture_height = config["height"]
    
    # Redimensiona a imagem para caber na textura existente
    cv_image = cv.resize(cv_image, (texture_width, texture_height), interpolation=cv.INTER_AREA)
    
    height, width, channels = cv_image.shape
    image_data = cv_image.flatten() / 255.0

    dpg.configure_item("image_window", width=width+20, height=height+40)
    dpg.configure_item("opencv_image_display", width=width, height=height)
    dpg.configure_item("opencv_image", width=width, height=height)
    update_texture()

def opencv_image_click_callback(sender, app_data, user_data):
    global mouse_pos
    mouse_pos = dpg.get_mouse_pos()
    update_texture()

# Carrega a imagem inicial
image_path = "res/imagem2.jpg"
#image_path = "res/civilization_8rcz.jpg"
cv_original_image = cv.imread(image_path)
cv_image = cv_original_image.copy()
cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGBA)  # Converte BGR para RGBA

# Pega dimensões da imagem
height, width, channels = cv_image.shape

# Preapara a imagem para Dear PyGui
image_data = cv_image.flatten() / 255.0  # Normalização para [0.0, 1.0]

# Configura a captura de vídeo (câmera)
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera or video file")
    exit()
# Define dimensões de captura da câmera
capture.set(cv.CAP_PROP_FRAME_WIDTH, width_cam)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, height_cam)

# Cria o contexto Dear PyGui
dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

# Cria o item handler para clique na imagem
with dpg.item_handler_registry(tag="opencv_image_handler"):
    dpg.add_item_clicked_handler(callback=opencv_image_click_callback)

# Cria o diálogo de seleção de arquivo
with dpg.file_dialog(directory_selector=False, show=False, callback=select_image_callback, tag="file_dialog_id", width=400, height=300):
    dpg.add_file_extension(".jpg,.png,.jpeg", color=(150, 255, 150, 255))

# Cria texturas dinâmicas de imagem e vídeo
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width, height, image_data, tag="opencv_image")
    dpg.add_dynamic_texture(width_cam, height_cam, image_data, tag="opencv_video")

# Cria a janela de efeitos
with dpg.window(label="Configurações",no_close=True,no_resize=True,no_collapse=True, width=effects_window_width, height=effects_window_height, no_move=True, pos=(0, 0), tag="effects_window"):
    dpg.add_combo(label="Modo de Software", items=["Imagem", "Câmera"], default_value=("Câmera" if software_mode == "camera" else "Imagem"), tag="image_camera_selector")
    with dpg.group(horizontal=True):
        dpg.add_button(label="Salvar Imagem", callback=save_image_callback)
        dpg.add_button(label="Selecionar Imagem", callback=lambda: dpg.show_item("file_dialog_id"))
    dpg.add_separator()
    dpg.add_combo(label="Efeito", items=effects_list_names, tag="effect_selector", callback=add_effect_callback)
    dpg.add_separator()

# Janela da imagem
with dpg.window(label="Imagem",no_close=True,no_resize=False,no_collapse=True, tag="image_window", pos=(effects_window_width + 1, 0)):
    dpg.add_image("opencv_image", tag="opencv_image_display")

# Janela do vídeo
with dpg.window(label="Vídeo",no_close=True,no_resize=True,no_collapse=True, tag="video_window", pos=(effects_window_width + 1, 0)):
    dpg.add_image("opencv_video", tag="opencv_video_display")

dpg.set_item_callback("image_camera_selector", change_mode_callback)
dpg.bind_item_handler_registry("opencv_image_display", "opencv_image_handler")
dpg.bind_item_handler_registry("opencv_video_display", "opencv_image_handler")

# Define tamanho padrão da janela
dpg.set_viewport_width(window_size_width) 
dpg.set_viewport_height(window_size_height)

# Mostra a viewport
dpg.show_viewport()

# Inicializa o modo correto
change_mode_callback()

try:
    # Loop de execução
    while dpg.is_dearpygui_running():
        update_video_frame()
        dpg.render_dearpygui_frame()
except SystemExit:
    # Trata finalização pelo fechamento da janela
    pass
finally:
    # Finaliza
    capture.release()
    dpg.destroy_context()
