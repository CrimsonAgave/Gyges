import flet as ft

import os
import send2trash
import pyperclip
import sys
import copy
import pickle

import numpy as np
import math
import random
import time

from typing import Union, List
import ftfy, html, re
import PIL

import huggingface_hub
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature
import torch
import faiss


CONFIG_DIR_NAME = "./config" 
CONFIG_PATH = CONFIG_DIR_NAME + "/config.pkl"
SOURCE_DIR_NAME = "./src"
HISTORY_PATH = SOURCE_DIR_NAME + "/history.pkl"
PATHES_PATH = SOURCE_DIR_NAME + "/paths.pkl"
MARKS_PATH = SOURCE_DIR_NAME + "/marks.pkl"
VECTRORS_PATH = SOURCE_DIR_NAME + "/vectors.pkl"

SIDEBAR_MIN_WIDTH = 72
SIDEBAR_HALF_PADDING = 5
TITLEBAR_HEIGHT = 40
WindowControlButton_WIDTH = 40
TOOLBAR_HEIGHT = 64
TOOLBAR_PADDING = 4
ICON_SIZE = 24
MAX_NAME_LENGETH = 8

paths = {}
config = {}
history = {}
marks = {}
vectors = {}
loading = False
logined = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

MODEL_PATH = "stabilityai/japanese-stable-clip-vit-l-16"


def check_format(path, formats):
    return (os.path.splitext(path)[1] in formats)

def path2filename(path):
    return os.path.basename(path)

def attach_a_unit(num, deciaml_places=2, unit="B"):
    suffixes = ("", "K", "M", "G", "T", "P", "E")
    i = int(math.floor(math.log(num, 1000))) if(num > 0) else 0
    rounded_num = round(num / 1000 ** i, deciaml_places)
    
    return f"{rounded_num} {suffixes[i]}{unit}"

def save_config():
    global config
    if(not os.path.isdir(CONFIG_DIR_NAME)):
        os.mkdir(CONFIG_DIR_NAME)
    with open(CONFIG_PATH, "wb") as f:
        pickle.dump(config, f)

def save_paths():
    global paths
    if(not os.path.isdir(SOURCE_DIR_NAME)):
        os.mkdir(SOURCE_DIR_NAME)
    with open(PATHES_PATH, "wb") as f:
        pickle.dump(paths, f)

def save_history():
    global history
    if(not os.path.isdir(SOURCE_DIR_NAME)):
        os.mkdir(SOURCE_DIR_NAME)
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)

def save_marks():
    global marks
    if(not os.path.isdir(SOURCE_DIR_NAME)):
        os.mkdir(SOURCE_DIR_NAME)
    with open(MARKS_PATH, "wb") as f:
        pickle.dump(marks, f)

def save_vectors():
    global vectors
    if(not os.path.isdir(SOURCE_DIR_NAME)):
        os.mkdir(SOURCE_DIR_NAME)
    with open(VECTRORS_PATH, "wb") as f:
        pickle.dump(vectors, f)

def recursive_get_picture_path(path, valid_path_set, formats, pb, pv):
    if(os.path.isdir(path)):
        files = os.listdir(path)
        pathes = [path + os.path.sep + file for file in files if(os.path.isdir(path + os.path.sep + file) or check_format(path + os.path.sep + file, formats))]
        if(len(pathes) != 0):
            pv /= len(pathes)
        else:
            pb.value += pv
            pb.update()
        for p in pathes:
            recursive_get_picture_path(p, valid_path_set, formats, pb, pv)
    else:
        valid_path_set.append(path)
        pb.value += pv
        pb.update()

def get_picture_path(path, formats, pb):
    global paths
    valid_path_set = []
    recursive_get_picture_path(path, valid_path_set, formats, pb, 1.0)
    paths["root_directory"] = path
    paths["HOME"] = valid_path_set
    save_paths()

def get_model():
    global model, tokenizer, processor
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
    processor = AutoImageProcessor.from_pretrained(
        MODEL_PATH
    )

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def tokenize(
    tokenizer,
    texts: Union[str, List[str]],
    max_seq_len: int = 77,
):
    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = tokenizer(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )



class WindowControlButton(ft.IconButton):
    def __init__(self, *args, tooltip, **kwargs):
        super().__init__(*args, tooltip=tooltip, **kwargs)
        self.height = 30
        self.width = WindowControlButton_WIDTH
        self.icon_size=16


class TabBarColumn(ft.Column):
    def __init__(self, page: ft.Page, layers_clear_func, black_curtain, set_gallery_visiblility):
        super().__init__()
        self.HEIGHT = TITLEBAR_HEIGHT
        self.ICON_SIZE = 20
        self.ICON_COLOR = ft.colors.ON_PRIMARY
        self.TEXT = {
            "CLOSE_TOOLTIP": "閉じる", 
            "MAXIMIZE_TOOLTIP": "最大化", 
            "MAXIMIZE_UNDO_TOOLTIP": "元に戻す（縮小）", 
            "MINIMIZE_TOOLTIP": "最小化",
            "LAYERS_CLEAR": "タブをすべて削除",          
            }

        self.spacing = 0
        self.margin = 0
        self.reset_gallery_scroll = lambda _: print("ERROR: The function of reset_gallery_scroll is not set")
        self.layers_clear_func = layers_clear_func
        self.black_curtain = black_curtain

        self.set_gallery_visiblility = set_gallery_visiblility

        self.selected_index = -1
        self.page = page
        self.tabs = []
        self.tabbar_content = []
        self.extended = 0

        self.tabbar = ft.Tabs(
            selected_index=0,
            animation_duration = 60,
            tab_alignment = ft.TabAlignment.START,
            label_color = ft.colors.ON_PRIMARY_CONTAINER,
            unselected_label_color=self.ICON_COLOR,
            tabs=self.tabs,
            on_change=self.on_change_tab,
        )

        self.content = ft.Stack(
            self.tabbar_content,
        )

        self.window_control_buttons = ft.Row(
            [
                self.get_titlebar_content()
            ],
            alignment=ft.MainAxisAlignment.END,
        )

        self.tab_drag_row = ft.Row(
            [
                ft.Container(self.tabbar),
                ft.WindowDragArea(ft.Container(height=48), expand=True),
            ],
        )

        self.controls = [
            ft.Stack(
                [
                    ft.Container(height=TITLEBAR_HEIGHT, expand=True, bgcolor=ft.colors.PRIMARY),
                    self.tab_drag_row,
                    self.window_control_buttons,
                ],
            ),
            ft.Container(self.content, expand=True), 
        ]

    def get_titlebar_content(self):
        self.layers_clear = WindowControlButton(
            icon=ft.icons.LAYERS_CLEAR_OUTLINED,
            icon_color=self.ICON_COLOR,
            tooltip=self.TEXT["LAYERS_CLEAR"],
            on_click=self.layers_clear_func,
            style = ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(),
                color=ft.colors.ON_BACKGROUND,
                overlay_color=ft.colors.SURFACE_VARIANT,
            )
        )

        self.minimize = WindowControlButton(
            icon=ft.icons.MINIMIZE,
            icon_color=self.ICON_COLOR,
            tooltip=self.TEXT["MINIMIZE_TOOLTIP"],
            on_click=self.window_minimize,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(),
                color=ft.colors.ON_BACKGROUND,
                overlay_color=ft.colors.SURFACE_VARIANT,
            )
        )

        self.maximize = WindowControlButton(
            icon=ft.icons.CROP_SQUARE,
            icon_color=self.ICON_COLOR,
            tooltip=self.TEXT["MAXIMIZE_TOOLTIP"],
            on_click=self.window_maximize,
            style = ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(),
                color=ft.colors.ON_BACKGROUND,
                overlay_color=ft.colors.SURFACE_VARIANT,
            )
        )

        self.close = WindowControlButton(
            icon=ft.icons.CLOSE,
            icon_color=self.ICON_COLOR,
            tooltip=self.TEXT["CLOSE_TOOLTIP"],
            on_click=lambda _: self.page.window_close(),
            style = ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(),
                color=ft.colors.ON_BACKGROUND,
                overlay_color=ft.colors.ERROR,
            )
        )

        return ft.Row(
            [
                self.layers_clear,
                self.minimize,
                self.maximize,
                self.close,
            ],
            spacing=0,
        )

    def window_minimize(self, e):
        self.page.window_minimized = True
        self.page.update()

    def window_maximize(self, e):
        global config
        self.page.window_maximized = not self.page.window_maximized
        if(self.page.window_maximized):
            self.maximize.icon = ft.icons.FILTER_NONE
            self.maximize.tooltip = self.TEXT["MAXIMIZE_UNDO_TOOLTIP"]
        else:
            self.maximize.icon = ft.icons.CROP_SQUARE
            self.maximize.tooltip = self.TEXT["MAXIMIZE_TOOLTIP"]
        
        config["window_maximized"] = self.page.window_maximized
        save_config()  

        self.page.update()        

    def make_new_tab(self, tab_type, s, content):
        self.add_to_history(tab_type, s)
        tab_content = []

        if(tab_type == "HOME"):
            icon = ft.Icon(name=ft.icons.HOME, size=self.ICON_SIZE)
            text = ""
        elif(tab_type == "SEARCH"):
            icon = ft.Icon(name=ft.icons.SEARCH, size=self.ICON_SIZE)
            text = s
        elif(tab_type == "IMAGE"):
            icon = ft.Icon(name=ft.icons.IMAGE, size=self.ICON_SIZE)
            text = path2filename(s)
        elif(tab_type == "HISTORY"):
            icon = ft.Icon(name=ft.icons.HISTORY, size=self.ICON_SIZE)
            text = path2filename(s)
        elif(tab_type == "ALBUM"):
            icon = ft.Icon(name=ft.icons.PHOTO_ALBUM, size=self.ICON_SIZE)
            text = s
        else:
            icon = ft.Icon(name=ft.icons.ERROR, size=self.ICON_SIZE)
            text = "ERROR"

        if(len(text) > MAX_NAME_LENGETH):
            text = text[:MAX_NAME_LENGETH-1] + "…"

        tab_content += [icon]
        if(len(text) > 0):
            tab_content += [ft.Text(text)]

        tab = ft.Tab(
            tab_content=ft.Row(tab_content),
        )
        tab.tab_type = tab_type
        self.tabbar.tabs.append(tab)
        self.tabbar_content.append(content)
        self.tabbar.selected_index = len(self.tabbar.tabs)-1
        self.reset_visibility(self.tabbar.selected_index)

        self.page.update()

    def add_to_history(self, tab_type, s):
        global history
        if(tab_type == "IMAGE"):
            history["IMAGE"].append(s)
        elif(tab_type == "SEARCH"):
            history["SEARCH"].append(s)
        save_history()

    def close_tab(self, tab_index):
        self.tabbar.selected_index = 0
        del self.tabbar.tabs[tab_index]
        del self.tabbar_content[tab_index]

        self.reset_visibility(self.tabbar.selected_index)
        self.page.update()

        for i in range(tab_index, len(self.tabbar.tabs)):
            self.tabbar_content[i].tab_index -= 1
            if(type(self.tabbar_content[i]) is SearchTab):
                self.tabbar_content[i].toolbar.tab_index -= 1  

    def close_last_tab(self):
        last_index = len(self.tabbar.tabs)-1
        if(self.tabbar.selected_index >= last_index):
            self.tabbar.selected_index = last_index-1
        del self.tabbar.tabs[last_index]
        del self.tabbar_content[last_index]
        self.reset_visibility(last_index-1)

    def on_change_tab(self, e):
        global loading
        index = e.control.selected_index
        self.reset_visibility(index)        
        
    def reset_visibility(self, index):
        tab_type = self.tabbar.tabs[index].tab_type

        self.set_gallery_visiblility(tab_type)
        self.black_curtain.visible = (tab_type == "IMAGE")
        #self.reset_gallery_scroll(tab_type)

        for i in range(len(self.tabbar_content)):
            self.tabbar_content[i].visible = (i == index)
            if(self.tabbar.tabs[i].tab_type == "HISTORY"):
                self.tabbar_content[i].history_gallery.visible = (i == index)
            elif(self.tabbar.tabs[i].tab_type == "ALBUM"):
                self.tabbar_content[i].album_gallery.visible = (i == index)

        self.page.update()

    def change_tab_display(self, tab_index, file_name):
        text = path2filename(file_name)
        if(len(text) > MAX_NAME_LENGETH):
            text = text[:MAX_NAME_LENGETH-1] + "…"
        self.tabbar.tabs[tab_index].tab_content.controls[1].value = text


class ImageTab(ft.Container):
    def __init__(self, tab_index, close_func, page: ft.Page, path, valid_path_list, change_tab_display, set_bookmark_image):
        super().__init__()
        global paths, marks
        self.TEXT = {
            "CLOSE_TOOLTIP": "閉じる",
            "BACK_TOOLTIP": "戻る",
            "IS_IN_ALBUM_TOOLTIP": "アルバムに登録さています。",
            "BOOKMARK_TOOLTIP": "ブックマーク",
            "BOOKMARKED": "この画像をブックマークしました。",
            "ALBUM_TOOLTIP": "アルバム",
            "REMOOVE_BOOKMARK": "画像のブックマークを外しました。",
            "MORE_VERT_TOOLTIP": "メニュー",
            "GET_PATH_TOOLTIP": "画像のアドレスをコピー",
            "GOT_PATH": "画像のアドレスをクリップボードにコピーしました。",

            "ALBUM": "アルバム「",
            "ADDED": "に登録されています。",
            "PUT_IN_ALBUM": "」に登録しました。",
            "REMOVE_FROM_ALBUM": "」への登録を解除しました。",
            "NEW_ALBUM": "新しいアルバム",
            "MAKE_NEW_ALBUM": "新しいアルバムを作成",
            "NEW_ALBUM_TITLE": "アルバムタイトル",
            "CONFIRM": "決定",
            "INVALID_TITLE": "そのタイトルは使えません。",
            "THIS_TITLE_ALREADY_EXISTS": "既存のアルバムと同じタイトルです。別のタイトルを使用してください。",
            "MAKED_ALBUM_AND_ADDED": "」を作成し、この画像を登録しました。",

            "DELETE": "画像の削除",
            "DELETE_IT?": "本当にこの画像を削除しますか？",
            "DELETED": "画像をゴミ箱に入れました。",
            "YES": "はい",
            "CANCEL": "キャンセル",
        }

        self.tab_index = tab_index
        self.page = page
        self.close_func = close_func
        self.valid_path_list = valid_path_list
        self.path_idx = self.valid_path_list.index(path)
        self.path = path
        self.change_tab_display = change_tab_display
        self.set_bookmark_image = set_bookmark_image

        self.bgcolor = ft.colors.BLACK
        self.icon_color = ft.colors.WHITE
        self.width = 4096
        self.height = 2160
        
        self.menu_opened = True
        self.expanded_above = False
        self.expanded_below = False

        self.get_shadow()
        self.image_scale = ft.Scale(scale_x=1, scale_y=1)
        self.scroll_one_delta = 133

        self.image = ft.Container(
            image_src=self.path,
            image_fit=ft.ImageFit.CONTAIN,
            alignment=ft.alignment.center,
            scale=self.image_scale,
            width=self.page.window_width,
            height=self.page.window_height - TITLEBAR_HEIGHT,
            left=0,
            top=0,
        )

        self.isin_bookmark = (self.path in marks["BOOKMARK"])
        
        self.name_label = self.get_name_label()
        self.name_label.visible = self.menu_opened

        self.menu = self.get_menu()
        self.menu.visible = self.menu_opened

        gd_boxes = self.get_gd_boxes()
        self.gd_icon_boxes = self.get_gd_icon_boxes()
        self.gd_icon_boxes.visible = self.menu_opened

        self.more_content = ft.Container(bgcolor=ft.colors.GREEN, height=2000)

        self.tab_contents = [
                self.image,
                self.name_label,
                self.gd_icon_boxes,
                gd_boxes,
                self.menu,
            ]
        self.main_content = ft.Stack(self.tab_contents)
        
        self.content = self.main_content
        """ft.Column(
            [
                self.main_content,
                #self.more_content,
            ]
        )    """ 

    def get_menu(self):
        global paths
        close_image_tab_icon_button = ft.IconButton(
            icon=ft.icons.CLOSE,
            icon_size = ICON_SIZE,
            icon_color = self.icon_color,
            tooltip=self.TEXT["CLOSE_TOOLTIP"],
            on_click=self.close_tab,
        )

        back_icon_button = ft.IconButton(
            icon=ft.icons.ARROW_BACK,
            icon_size = ICON_SIZE,
            icon_color = self.icon_color,
            tooltip=self.TEXT["BACK_TOOLTIP"],
            on_click=self.back_tab,
        )

        self.isin_album_icon_button = ft.IconButton(
            icon=ft.icons.STAR,
            icon_size = ICON_SIZE,
            icon_color = self.icon_color,
            tooltip=self.TEXT["IS_IN_ALBUM_TOOLTIP"],
            on_click=self.click_isin_album,
            visible=False,
        )

        if(0 < sum(self.path in album_paths for album_paths in marks["ALBUM"].values())):
            self.isin_album_icon_button.visible = True
        else:
            self.isin_album_icon_button.visible = False

        if(self.isin_bookmark):
            bookmark_icon = ft.icons.BOOKMARK
        else:
            bookmark_icon = ft.icons.BOOKMARK_BORDER

        self.bookmark_icon_button = ft.IconButton(
            icon=bookmark_icon,
            icon_size = ICON_SIZE,
            icon_color = self.icon_color,
            tooltip=self.TEXT["BOOKMARK_TOOLTIP"],
            on_click=self.click_bookmark,
        )

        self.album_icon_button = self.get_album_menu()

        self.delete_popup = ft.AlertDialog(
            modal=True,
            content=ft.Text(self.TEXT["DELETE_IT?"]),
            actions=[
                ft.TextButton(self.TEXT["YES"], on_click=self.delete_image),
                ft.TextButton(self.TEXT["CANCEL"], on_click=self.close_delete_popup),
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.delete_snap_bar = ft.SnackBar(ft.Text(self.TEXT["DELETED"]))
        self.trash_icon_button = ft.IconButton(
            icon=ft.icons.DELETE_OUTLINED,
            icon_size = ICON_SIZE,
            icon_color = self.icon_color,
            tooltip=self.TEXT["DELETE"],
            on_click=self.open_delete_popup,
        )

        pupup_menu_icon_button = ft.PopupMenuButton(
            items=[
                ft.PopupMenuItem(text=self.TEXT["GET_PATH_TOOLTIP"], on_click=self.get_path),
            ],
            content=ft.Icon(name=ft.icons.MORE_VERT, color=self.icon_color),
            tooltip=self.TEXT["MORE_VERT_TOOLTIP"],
        )

        space_container = ft.Container(expand=True)

        menu_content = ft.Row(
            [
                close_image_tab_icon_button,
                back_icon_button,
                space_container,
                self.isin_album_icon_button,
                self.bookmark_icon_button,
                ft.Container(self.album_icon_button, padding=10),
                self.trash_icon_button,
                pupup_menu_icon_button,
            ],
        )

        menu = ft.Stack(
            [
                self.top_shadow,
                ft.Container(
                    content = menu_content,
                    margin=ft.margin.only(top=ICON_SIZE // 2, bottom=ICON_SIZE // 2, left=2*ICON_SIZE, right=2*ICON_SIZE),
                    height=2*ICON_SIZE,
                ),
            ]
        )

        return menu        

    def get_album_menu(self):
        global marks
        self.popup_album_menu_items = [] 
        for i, (album_title, album_paths) in enumerate(marks["ALBUM"].items()):
            isin_album_icon = ft.Icon(name=ft.icons.STAR)
            if(self.path in album_paths):
                isin_album_icon.visible = True
            else:
                isin_album_icon.visible = False
            content = ft.Row(
                [
                    ft.Text(album_title),
                    ft.Container(expand=True),
                    isin_album_icon,
                ]
            )
            item = ft.PopupMenuItem(content=content, on_click=lambda e, album_title=album_title: self.put_in_album(e, album_title))
            self.popup_album_menu_items.append(item)

        for i, album_title in enumerate(marks["ALBUM"].keys()):
            if(self.path in marks["ALBUM"][album_title]):
                self.popup_album_menu_items[i].content.controls[2].visible = True
            else:
                self.popup_album_menu_items[i].content.controls[2].visible = False

        last_item = ft.PopupMenuItem(
            content=ft.Row(
                [
                    ft.Icon(name=ft.icons.ADD),
                    ft.Container(expand=True),
                    ft.Text(self.TEXT["NEW_ALBUM"]),
                ]
            ),
            on_click=self.popup_new_album_dialog,
        )

        self.popup_album_menu_items.append(last_item)

        album_menu_button = ft.PopupMenuButton(
            items=self.popup_album_menu_items,
            content=ft.Icon(name=ft.icons.PHOTO_ALBUM, color=self.icon_color),
            tooltip=self.TEXT["ALBUM_TOOLTIP"],
        )

        self.album_title_textfield = ft.TextField(
                label=self.TEXT["NEW_ALBUM_TITLE"],
                border=ft.InputBorder.UNDERLINE,
                filled=True,
        )

        self.new_album_dialog = ft.AlertDialog(
            modal=False,
            title=ft.Text(self.TEXT["MAKE_NEW_ALBUM"]),
            content=self.album_title_textfield,
            actions=[
                ft.TextButton(self.TEXT["CONFIRM"], on_click=self.make_new_album),
                ft.TextButton(self.TEXT["CANCEL"], on_click=self.close_new_album_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.invalid_title_snackbar = ft.SnackBar(ft.Text("ERROR"))
        self.new_album_added_snackbar = ft.SnackBar(ft.Text("ERROR"))

        return album_menu_button

    def click_bookmark(self, e):
        global marks
        self.isin_bookmark = not self.isin_bookmark

        if(self.isin_bookmark):
            marks["BOOKMARK"].append(self.path)
            bookmark_icon = ft.icons.BOOKMARK
            self.page.snack_bar = ft.SnackBar(ft.Text(self.TEXT["BOOKMARKED"]))
            
        else:
            marks["BOOKMARK"].remove(self.path)
            bookmark_icon = ft.icons.BOOKMARK_BORDER
            self.page.snack_bar = ft.SnackBar(ft.Text(self.TEXT["REMOOVE_BOOKMARK"]))

        self.bookmark_icon_button.icon = bookmark_icon
        self.page.snack_bar.open = True
        self.page.update()

        save_marks()
        self.set_bookmark_image()
        self.page.update()

    def popup_new_album_dialog(self, e):
        self.page.dialog = self.new_album_dialog
        self.new_album_dialog.open = True
        self.page.update()

    def close_new_album_dialog(self, e):
        self.album_title_textfield.value = ""
        self.new_album_dialog.open = False
        self.page.update()

    def make_new_album(self, e):
        global marks
        if(self.album_title_textfield.value in marks["ALBUM"].keys()):
            self.invalid_title_snackbar.content = ft.Text(self.TEXT["THIS_TITLE_ALREADY_EXISTS"])
            self.page.snack_bar = self.invalid_title_snackbar
            self.page.snack_bar.open = True
            self.page.update()
        elif(self.album_title_textfield.value == ""):
            self.invalid_title_snackbar.content = ft.Text(self.TEXT["INVALID_TITLE"])
            self.page.snack_bar = self.invalid_title_snackbar
            self.page.snack_bar.open = True
            self.page.update()
        else:
            new_album_title = self.album_title_textfield.value
            self.album_title_textfield.value = ""
            self.new_album_dialog.open = False

            self.new_album_added_snackbar.content = ft.Text(self.TEXT["ALBUM"] + new_album_title + self.TEXT["MAKED_ALBUM_AND_ADDED"])
            self.page.snack_bar = self.new_album_added_snackbar
            self.page.snack_bar.open = True

            marks["ALBUM"][new_album_title] = [self.path]
            save_marks()

            content = ft.Row(
                [
                    ft.Text(new_album_title),
                    ft.Container(expand=True),
                    ft.Icon(name=ft.icons.STAR),
                ]
            )
            item = ft.PopupMenuItem(content=content, on_click=lambda e: self.put_in_album(e, new_album_title))
            self.popup_album_menu_items.insert(-1, item)

            self.isin_album_icon_button.visible = True

            self.page.update()

    def put_in_album(self, e, album_title):
        global config, marks
        if(not album_title in marks["ALBUM"].keys()):
            print("ERROR: such album is not found", file=sys.stderr)
            exit(1)
        
        if(not self.path in marks["ALBUM"][album_title] ):
            marks["ALBUM"][album_title].append(self.path)
            self.page.snack_bar = ft.SnackBar(ft.Text(self.TEXT["ALBUM"] + str(album_title) + self.TEXT["PUT_IN_ALBUM"]))

            album_index = list(marks["ALBUM"].keys()).index(album_title)
            self.popup_album_menu_items[album_index].content.controls[2].visible = True

            self.isin_album_icon_button.visible = True
        else:
            album_image_index = marks["ALBUM"][album_title].index(self.path)
            del marks["ALBUM"][album_title][album_image_index]
            self.page.snack_bar = ft.SnackBar(ft.Text(self.TEXT["ALBUM"] + str(album_title) + self.TEXT["REMOVE_FROM_ALBUM"]))
            
            album_index = list(marks["ALBUM"].keys()).index(album_title)
            self.popup_album_menu_items[album_index].content.controls[2].visible = False

            isin_any_albums = sum((self.path in marks["ALBUM"][album_title]) for album_title in marks["ALBUM"].keys())
            if(isin_any_albums <= 0):
                self.isin_album_icon_button.visible = False
        save_marks()
        self.page.snack_bar.open = True
        self.page.update()

    def click_isin_album(self, e):
        global maks
        album_titles = []
        album_str = ""
        for album_title in marks["ALBUM"].keys():
            if(self.path in marks["ALBUM"][album_title]):
                album_titles.append(album_title)
        for i, album_title in enumerate(album_titles):
            if(i > 0):
                album_str += "「"
            album_str += album_title + "」" 
            
        self.isin_album_snackbar = ft.SnackBar(
            content=ft.Text(self.TEXT["ALBUM"] + album_str + self.TEXT["ADDED"])
        )
        self.page.snack_bar = self.isin_album_snackbar
        self.page.snack_bar.open = True
        self.page.update()

    def get_shadow(self):
        self.top_shadow = ft.Container(
            height=3*ICON_SIZE,
            gradient=ft.LinearGradient(
                begin=ft.alignment.top_center,
                end=ft.alignment.bottom_center,
                colors=[ft.colors.with_opacity(0.5, self.bgcolor), ft.colors.with_opacity(0, self.bgcolor)],
            ),
            expand=True,
        )

        self.bottom_shadow = ft.Container(
            height=24,
            gradient=ft.LinearGradient(
                begin=ft.alignment.top_center,
                end=ft.alignment.bottom_center,
                colors=[ft.colors.with_opacity(0, self.bgcolor), ft.colors.with_opacity(0.5, self.bgcolor)],
            ),
            expand=True,
        )

    def get_gd_boxes(self):
        l_gd = ft.GestureDetector(
            mouse_cursor=ft.MouseCursor.CLICK,
            drag_interval=40,
            on_tap=self.on_click_left,
            on_scroll=self.on_scroll,
            on_pan_update=self.on_image_pan_update,
        )

        c_gd = ft.GestureDetector(
            mouse_cursor=ft.MouseCursor.BASIC,
            drag_interval=40,
            on_tap=self.on_click_center,
            on_scroll=self.on_scroll,
            on_pan_update=self.on_image_pan_update,
        )

        r_gd = ft.GestureDetector(
            mouse_cursor=ft.MouseCursor.CLICK,
            drag_interval=40,
            on_tap=self.on_click_right,
            on_scroll=self.on_scroll,
            on_pan_update=self.on_image_pan_update,
        )

        l_gd_box = ft.Container(
            content=l_gd,
            width=2*SIDEBAR_MIN_WIDTH,
        )

        c_gd_box = ft.Container(
            content=c_gd,
            expand=True,
        )

        r_gd_box = ft.Container(
            content=r_gd,
            width=2*SIDEBAR_MIN_WIDTH,
        )
        
        return ft.Row(
                    [
                        l_gd_box,
                        c_gd_box,
                        r_gd_box
                    ],
                    expand=True,
                )
        
    def get_gd_icon_boxes(self):
        l_icon = ft.Icon(name=ft.icons.ARROW_CIRCLE_LEFT_OUTLINED, size=ICON_SIZE)
        r_icon = ft.Icon(name=ft.icons.ARROW_CIRCLE_RIGHT_OUTLINED, size=ICON_SIZE)

        l_icon_box = ft.Container(
            content=l_icon,
            width=2*SIDEBAR_MIN_WIDTH,
        )

        c_icon_box = ft.Container(
            content=None,
            expand=True,
        )

        r_icon_box = ft.Container(
            content=r_icon,
            width=2*SIDEBAR_MIN_WIDTH,
        )

        return ft.Row(
                    [
                        l_icon_box,
                        c_icon_box,
                        r_icon_box
                    ],
                    expand=True,
                )

    def get_name_label(self):
        name_text_container = ft.Container(
            content=ft.Text(self.path),
            alignment=ft.alignment.bottom_left,
            expand=True,
        )

        return ft.Stack(
            [
                ft.Container(
                    self.bottom_shadow,
                    alignment=ft.alignment.bottom_center,
                ),
                name_text_container,
            ]
        )
        
    def on_click_left(self, e):
        if(not self.expanded_above):
            if(self.path_idx - 1 >= 0):
                self.path_idx -= 1
                self.init_display(self.path_idx)

    def on_click_center(self, e):
        self.menu_opened = not self.menu_opened
        for i in range(len(self.tab_contents)):
            if(not i in [0, 3]):
                self.tab_contents[i].visible = self.menu_opened

        self.page.update()

    def on_click_right(self, e):
        if(not self.expanded_above):
            if(self.path_idx + 1 < len(self.valid_path_list)):
                self.path_idx += 1
                self.init_display(self.path_idx)

    def on_scroll(self, e):
        if(self.expanded_above == False and self.expanded_below == False):
            if(e.scroll_delta_y < 0):
                self.expanded_above = True
            elif(e.scroll_delta_y > 0):
                #self.expanded_below = True
                pass
        elif(self.expanded_above):
            self.on_scroll_above(e)
        elif(self.expanded_below):
            self.on_scroll_below(e)
        
    def on_scroll_above(self, e):
        if(self.image.scale.scale_x >= 1.0):
            tmp = self.scroll_one_delta * 2
            self.image.scale.scale_x -= e.scroll_delta_y / tmp
            self.image.scale.scale_y -= e.scroll_delta_y / tmp
        if(self.image.scale.scale_x < 1.0):
            self.image.scale.scale_x = 1.0
            self.image.scale.scale_y = 1.0
            self.expanded_above = False
            self.expanded_below = False
            self.image.top = 0
            self.image.left = 0  
        self.page.update()

    def on_image_pan_update(self, e):
        if(self.expanded_above):
            self.image.top = self.image.top + e.delta_y
            self.image.left = self.image.left + e.delta_x  
        self.page.update()

    def on_scroll_below(self, e):
        pass

    def init_display(self, path_idx):
        global history, marks
        self.path_idx = path_idx
        self.path = self.valid_path_list[self.path_idx]
        self.image.image_src = self.valid_path_list[self.path_idx]
        self.isin_bookmark = (self.path in marks["BOOKMARK"])

        if(self.isin_bookmark):
            bookmark_icon = ft.icons.BOOKMARK
        else:
            bookmark_icon = ft.icons.BOOKMARK_BORDER
        self.bookmark_icon_button.icon = bookmark_icon

        for i, album_title in enumerate(marks["ALBUM"].keys()):
            if(self.path in marks["ALBUM"][album_title]):
                self.popup_album_menu_items[i].content.controls[2].visible = True
            else:
                self.popup_album_menu_items[i].content.controls[2].visible = False

        if(0 < sum(self.path in album_paths for album_paths in marks["ALBUM"].values())):
            self.isin_album_icon_button.visible = True
        else:
            self.isin_album_icon_button.visible = False


        self.tab_contents[1] = self.get_name_label()
        self.tab_contents[1].visible = self.menu_opened
        self.change_tab_display(self.tab_index, self.path)

        history["IMAGE"].append(self.path)
        self.page.update()

    def close_tab(self, e, deleted=False):
        self.close_func(self.tab_index, deleted)

    def back_tab(self, e):
        if(0):
            pass
        else:
            self.close_tab(e)

    def open_delete_popup(self, e):
        self.page.dialog = self.delete_popup
        self.delete_popup.open = True
        self.page.update()
    
    def close_delete_popup(self, e):
        self.delete_popup.open = False
        self.page.update()

    def delete_image(self, e):
        self.delete_popup.open = False
        self.close_tab(e, deleted=True)

    def get_path(self, e):
        pyperclip.copy(self.path)
        self.page.snack_bar = ft.SnackBar(ft.Text(self.TEXT["GOT_PATH"]))
        self.page.snack_bar.open = True
        self.page.update()


class SearchTab(ft.Container):
    def __init__(self, tab_index, page, query, search_text_submit, close_func, reset_gallery, open_browsing_history, open_image):
        super().__init__()
        self.page = page
        self.query = query
        self.tab_index = tab_index
        self.search_text_submit = search_text_submit
        self.close_func = close_func

        self.toolbar = ToolBar("SEARCH", self.page, None, self.search_text_submit, self.close_func, self.tab_index, reset_gallery, open_browsing_history)
        self.toolbar.search_text.value = query
        self.toolbar.search_text.disabled = True

        image_type = "SEARCH"
        self.search_gallery = Gallery(self.page, open_image, image_type)

        self.content = ft.Column(
            [
                self.toolbar,
                self.search_gallery
            ]
        )


class BrowsingHistoryTab(ft.Container):
    def __init__(self, page, tab_index, close_func, open_browsing_history, history_gallery):
        super().__init__()
        self.TEXT = {
            "FEATURE_OF_IMAGE": "履歴内の画像の特徴",
            "DELETE_HISTORY": "閲覧履歴の削除",
            "DELETE_THEM_ALL?": "閲覧履歴をすべて削除しますか？",
            "YES": "はい",
            "CANCEL": "キャンセル",
        }
        self.ICON_SIZE = 24
        self.ICON_COLOR = ft.colors.ON_BACKGROUND
        self.BGCOLOR = ft.colors.BACKGROUND

        self.page = page
        self.tab_index = tab_index
        self.close_func = close_func
        self.open_browsing_history = open_browsing_history

        open_image = history_gallery.open_image
        image_type = history_gallery.image_type
        self.history_gallery = Gallery(self.page, open_image, image_type)
        self.history_gallery.controls = copy.copy(history_gallery.controls)


        self.toolbar = ToolBar("HISTORY", self.page, None, self.search_text_submit, self.close_tab, self.tab_index, None, self.open_browsing_history)
        self.toolbar.search_text.label = self.TEXT["FEATURE_OF_IMAGE"]

        self.set_content()

    def set_content(self):
        trash_icon_button = ft.IconButton(
            icon=ft.icons.DELETE,
            tooltip=self.TEXT["DELETE_HISTORY"],
            on_click=self.open_delete_popup,
        )

        self.delete_dialog = ft.AlertDialog(
            modal=False,
            content=ft.Container(
                ft.Text(self.TEXT["DELETE_THEM_ALL?"]),
                padding=10,
            ),
            actions=[
                ft.TextButton(self.TEXT["YES"], on_click=self.delete_dialog),
                ft.TextButton(self.TEXT["CANCEL"], on_click=self.close_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )


        self.content = ft.Column(
            [
                self.toolbar,
                self.history_gallery,
            ]
        )

    def search_text_submit(self, e):
        print(e)

    def open_delete_popup(self, e):
        self.page.dialog = self.delete_dialog
        self.delete_dialog.open = True
        self.page.update()

    def delete_dialog(self, e):
        global history
        history["IMAGE"] = []
        save_history()
        self.history_gallery.set_pictures()
        self.delete_dialog.open = False
        self.page.update()

    def close_dialog(self, e):
        self.delete_dialog.open = False
        self.page.update()

    def close_tab(self, e):
        self.close_func(self.tab_index)


class ToolBar(ft.Container):
    def __init__(self, tab_type, page: ft.Page, sidebar, search_text_submit, close_func, tab_index, reset_all_gallery, open_browsing_history):
        super().__init__()
        self.BGCOLOR = ft.colors.BACKGROUND
        self.ICON_SIZE = 24
        self.ICON_COLOR = ft.colors.ON_BACKGROUND
        self.LEADING_WIDTH = 80
        self.RIGHT_END_WIDTH = 10
        self.TEXT = {
            "MENU_TOOLTIP" : "メインメニュー",
            "CLOSE_TOOLTIP": "タブを閉じる",
            "DISPLAY_SETTINGS": "表示設定",
            "HOME_DISPLAY_SETTINGS": "表示設定：ホーム",
            "BOOKMARK_DISPLAY_SETTINGS": "表示設定：ブックマーク",

            "SEARCH_TOOLTIP" : "連想検索",
            "FEATURE_OF_IMAGE": "画像の特徴",
            "POPUP_MENU_TOOLTIP": "メニュー",
            "POPUP_MENU_ITEM_1_TOOLTIP" : "閲覧履歴",

            "QUESTION": "使い方",
            "SORTING": "並び替え",
            "RANDOM": "ランダム",
            "ASCENDING": "古い順",
            "DESCENDING": "新しい順",

            "IMAGE_THUMBNAIL": "画像のサムネイル",

            "NONE": "NONE",
            "CONTAIN": "CONTAIN",
            "COVER": "COVER",
            "FILL":  "FILL",

            "SIZE": "画像の大きさ",

            "NUMBER": "画像の最大表示枚数\n(1000枚推奨)",

            "CONFIRM": "決定",
            "CANCEL": "キャンセル",
            "LOADING": "読み込み中",
            "WAITING": "お待ちください",
        }

        self.tab_type = tab_type
        self.height = TOOLBAR_HEIGHT
        self.page = page
        self.close_func = close_func
        self.tab_index = tab_index
        if(self.tab_type == "HOME"):
            self.sidebar = sidebar
        self.isopen = False
        self.bgcolor = self.BGCOLOR
        self.padding = ft.padding.only(left=SIDEBAR_HALF_PADDING, top=TOOLBAR_PADDING, right=TOOLBAR_PADDING, bottom=TOOLBAR_PADDING)
        
        self.reset_all_gallery = reset_all_gallery
        self.search_text_submit = search_text_submit
        self.open_browsing_history = open_browsing_history
        
        self.set_content()
                    
    def set_content(self):
        self.menu = ft.IconButton(
            icon=ft.icons.MENU,
            icon_color=self.ICON_COLOR,
            icon_size=self.ICON_SIZE,
            tooltip=self.TEXT["MENU_TOOLTIP"],
            on_click=self.toggle_menu,
        )

        self.close = ft.IconButton(
            icon=ft.icons.CLOSE,
            icon_color=self.ICON_COLOR,
            icon_size=self.ICON_SIZE,
            tooltip=self.TEXT["CLOSE_TOOLTIP"],
            on_click=self.close_tab,
        )

        self.set_display_settings()

        self.display_settings = ft.IconButton(
            icon=ft.icons.TUNE,
            icon_color=self.ICON_COLOR,
            icon_size=self.ICON_SIZE,
            tooltip=self.TEXT["DISPLAY_SETTINGS"],
            on_click=self.popup_display_settings,
        )

        self.popup_menu_icon = ft.Icon(
            name=ft.icons.MORE_VERT,
            color=self.ICON_COLOR,
            size=self.ICON_SIZE,
        )

        self.popup_menu = ft.PopupMenuButton(
            content=self.popup_menu_icon,
            items=[
                ft.PopupMenuItem(text=self.TEXT["POPUP_MENU_ITEM_1_TOOLTIP"], on_click=self.open_browsing_history),
            ],
            tooltip=self.TEXT["POPUP_MENU_TOOLTIP"],
        )

        self.question = ft.IconButton(
            icon=ft.icons.HELP_OUTLINED,
            icon_color=self.ICON_COLOR,
            icon_size=self.ICON_SIZE,
            tooltip=self.TEXT["QUESTION"],
        )

        self.search_text = ft.TextField(
                    prefix_icon=ft.icons.SEARCH,
                    border_color=ft.colors.ON_BACKGROUND,
                    label=self.TEXT["FEATURE_OF_IMAGE"],
                    on_submit=self.on_submit
                )
        
        tmp = ft.ResponsiveRow(
                [
                    ft.Container(col=3),
                    ft.Container(self.search_text, col=6),
                    ft.Container(col=3),
                ],
                spacing=0,
            )
        
        search_container = ft.Container(
            content=ft.Container(
                    content=tmp,
                ),
            alignment=ft.alignment.center,
            expand=True,
        )

        if(self.tab_type == "HOME"):
            main_icon_button = self.menu
        else:
            main_icon_button = self.close

        self.content = ft.Row(
            [
                ft.Container(
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Row([main_icon_button, self.display_settings]),
                                padding=ft.padding.only(left=20),
                            ),
                        ],
                    ),
                    margin=0,
                ),
                search_container,
                self.popup_menu,
                self.question,
                ft.Container(padding=ft.padding.only(right=self.RIGHT_END_WIDTH)),
            ],
        )

    def set_display_settings(self):
        self.sorting_dropdown = ft.Dropdown(
            options=[
                ft.dropdown.Option(self.TEXT["RANDOM"]),
                ft.dropdown.Option(self.TEXT["ASCENDING"]),                                    
                ft.dropdown.Option(self.TEXT["DESCENDING"]),                                    
            ],
            width=2*SIDEBAR_MIN_WIDTH,
        )

        first_row = ft.DataRow(
            cells=[
                ft.DataCell(ft.Text(self.TEXT["SORTING"])),
                ft.DataCell(self.sorting_dropdown),
            ]
        )

        self.fit_dropdown = ft.Dropdown(
            options=[
                ft.dropdown.Option(self.TEXT["NONE"]),
                ft.dropdown.Option(self.TEXT["CONTAIN"]),                                    
                ft.dropdown.Option(self.TEXT["COVER"]),                                    
                ft.dropdown.Option(self.TEXT["FILL"]),                                    
            ],
            width=2*SIDEBAR_MIN_WIDTH,
        )


        second_row = ft.DataRow(
            cells=[
                ft.DataCell(ft.Text(self.TEXT["IMAGE_THUMBNAIL"])),
                ft.DataCell(self.fit_dropdown),
            ]
        )

        self.size_slider = ft.Slider(min=100, max=1000, divisions=9, label="{value}", adaptive=True)

        third_row = ft.DataRow(
            cells=[
                ft.DataCell(ft.Text(self.TEXT["SIZE"])),
                ft.DataCell(self.size_slider),
            ]
        )

        self.num_slider = ft.Slider(min=1000, max=10000, divisions=9, label="{value}", adaptive=True)

        fourth_row = ft.DataRow(
            cells=[
                ft.DataCell(ft.Text(self.TEXT["NUMBER"])),
                ft.DataCell(self.num_slider),
            ]
        )

        self.set_settings_value()
        self.setting_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("")),
                ft.DataColumn(ft.Text(""), numeric=True),
            ],
            rows=[
                first_row,
                second_row,
                third_row,
                fourth_row,
            ],
            data_row_max_height=60
        )


        confirm_button = ft.TextButton(text=self.TEXT["CONFIRM"], on_click=self.confirm_display_settings)
        cancel_button = ft.TextButton(text=self.TEXT["CANCEL"], on_click=self.close_display_settings)

        self.display_settings_content = ft.Container(
            content=self.setting_table,
            width=6*SIDEBAR_MIN_WIDTH,
            height= 400
        )

        self.display_settings_popup = ft.AlertDialog(
            modal=False,
            title=ft.Text(self.TEXT["HOME_DISPLAY_SETTINGS"]),
            content=self.display_settings_content,
            on_dismiss=self.close_display_settings,
            actions=[
                confirm_button,
                cancel_button,
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.loading_popup = ft.AlertDialog(
            modal=True,
            title=ft.Text(self.TEXT["LOADING"]),
            content=ft.Container(
                ft.Column(
                    [
                        ft.Container(expand=True),
                        ft.Container(ft.ProgressRing(width=SIDEBAR_MIN_WIDTH, height=SIDEBAR_MIN_WIDTH, stroke_width=SIDEBAR_MIN_WIDTH/20), alignment=ft.alignment.center),
                        ft.Container(expand=True),
                        ft.Container(ft.Text(self.TEXT["WAITING"]), alignment=ft.alignment.center),
                    ],
                ),
                height=120,
            )
        )

    def toggle_menu(self, e):
        self.isopen = not self.isopen
        self.sidebar.rail.extended = self.isopen
        self.page.update()

    def close_tab(self, e):
        self.close_func(self.tab_index)
        
    def popup_display_settings(self, e):
        if(self.tab_type == "HOME" or self.tab_type == "BOOKMARK"):
            self.page.dialog = self.display_settings_popup

            self.display_settings_popup.open = True
            self.page.update()

    def set_sorting_droppdown_value(self):
        self.sorting_dropdown.disabled = False
        if(self.tab_type == "HOME"):
            self.sorting_dropdown.value = config["home_sorting"]
        elif(self.tab_type == "SEARCH"):
            self.sorting_dropdown.value = self.TEXT["DESCENDING"]
            self.sorting_dropdown.disabled = True
        elif(self.tab_type == "BOOKMARK"):
            self.sorting_dropdown.value = config["bookmark_sorting"]
        elif(self.tab_type == "HISTORY"):
            self.sorting_dropdown.value = config["history_sorting"]
        save_config()

    def set_fit_dropdown_value(self):
        if(self.tab_type == "HOME"):
            self.fit_dropdown.value = config["home_fit"]
        elif(self.tab_type == "SEARCH"):
            pass
        elif(self.tab_type == "BOOKMARK"):
            self.fit_dropdown.value = config["bookmark_fit"]
        elif(self.tab_type == "HISTORY"):
            self.fit_dropdown.value = config["history_fit"]
        save_config()

    def set_size_slider_value(self):
        if(self.tab_type == "HOME"):
            self.size_slider.value = config["home_size"]
        elif(self.tab_type == "SEARCH"):
            pass
        elif(self.tab_type == "BOOKMARK"):
            self.size_slider.value = config["bookmark_size"]
        elif(self.tab_type == "HISTORY"):
            self.size_slider.value = config["history_size"]
        save_config()

    def set_num_slider_value(self):
        if(self.tab_type == "HOME"):
            self.num_slider.value = config["home_num"]
        elif(self.tab_type == "SEARCH"):
            pass
        elif(self.tab_type == "BOOKMARK"):
            self.num_slider.value = config["bookmark_num"]
        elif(self.tab_type == "HISTORY"):
            self.num_slider.value = config["history_num"]
        save_config()

    def set_settings_value(self):
        self.set_sorting_droppdown_value()
        self.set_fit_dropdown_value()
        self.set_size_slider_value()
        self.set_num_slider_value()

    def confirm_display_settings(self, e):
        self.page.dialog = self.loading_popup
        self.loading_popup.open = True
        self.page.update()

        change_sorting = False
        change_fit = False
        change_size = False
        change_num = False

        # first_row - sorting
        if(self.tab_type == "HOME"):
            if(config["home_sorting"] == self.sorting_dropdown.value):
                if(self.sorting_dropdown.value == self.TEXT["RANDOM"]):
                    change_sorting = True
                else:
                    change_sorting = False
            else:
                change_sorting = True
                config["home_sorting"] = self.sorting_dropdown.value
                save_config()
                self.set_sorting_droppdown_value()
        elif(self.tab_type == "ROOT_FOLDER"):
            pass
        elif(self.tab_type == "BOOKMARK"):
            if(config["bookmark_sorting"] == self.sorting_dropdown.value):
                change_sorting = False
            else:
                change_sorting = True
                config["bookmark_sorting"] = self.sorting_dropdown.value
                save_config()
                self.set_sorting_droppdown_value()
        elif(self.tab_type == "HISTORY"):
            pass
        elif(self.tab_type == "ALBUM"):
            pass


        # second_row - fit
        if(self.tab_type == "HOME"):
            if(config["home_fit"] == self.fit_dropdown.value):
                change_fit = False
            else:
                change_fit = True
                config["home_fit"] = self.fit_dropdown.value
                save_config()
                self.set_fit_dropdown_value()
        elif(self.tab_type == "ROOT_FOLDER"):
            pass
        elif(self.tab_type == "BOOKMARK"):
            if(config["bookmark_fit"] == self.fit_dropdown.value):
                change_fit = False
            else:
                change_fit = True
                config["bookmark_fit"] = self.fit_dropdown.value
                save_config()
                self.set_fit_dropdown_value()
        elif(self.tab_type == "HISTORY"):
            pass
        elif(self.tab_type == "ALBUM"):
            pass


        # third_row - size
        if(self.tab_type == "HOME"):
            if(config["home_size"] == self.size_slider.value):
                change_size = False
            else:
                change_size = True
                config["home_size"] = self.size_slider.value
                save_config()
                self.set_size_slider_value()
        elif(self.tab_type == "ROOT_FOLDER"):
            pass
        elif(self.tab_type == "BOOKMARK"):
            if(config["bookmark_size"] == self.size_slider.value):
                change_size = False
            else:
                change_size = True
                config["bookmark_size"] = self.size_slider.value
                save_config()
                self.set_size_slider_value()
        elif(self.tab_type == "HISTORY"):
            if(config["history_size"] == self.size_slider.value):
                change_size = False
            else:
                change_size = True
                config["history_size"] = self.size_slider.value
                save_config()
                self.set_size_slider_value()
        elif(self.tab_type == "ALBUM"):
            pass

        # fourth_row - number
        if(self.tab_type == "HOME"):
            if(config["home_num"] == self.num_slider.value):
                change_num = False
            else:
                change_num = True
                config["home_num"] = int(self.num_slider.value)
                save_config()
                self.set_num_slider_value()
        elif(self.tab_type == "ROOT_FOLDER"):
            pass
        elif(self.tab_type == "BOOKMARK"):
            if(config["bookmark_num"] == self.num_slider.value):
                change_num = False
            else:
                change_num = True
                config["bookmark_num"] = int(self.num_slider.value)
                save_config()
                self.set_num_slider_value()
        elif(self.tab_type == "HISTORY"):
            if(config["history_num"] == self.num_slider.value):
                change_num = False
            else:
                change_num = True
                config["history_num"] = int(self.num_slider.value)
                save_config()
                self.set_num_slider_value()
        elif(self.tab_type == "ALBUM"):
            pass


        change_dict = {
            "sorting": change_sorting,
            "fit": change_fit,
            "size": change_size,
            "num": change_num,
        }
        
        self.reset_all_gallery(change_dict)
        self.loading_popup.open = False
        self.page.update()

    def close_display_settings(self, e):
        global config
        self.display_settings_popup.open = False

        self.page.update()

    def on_submit(self, e):
        self.search_text_submit(e)
        self.search_text.value = ""
        self.page.update()


class SideBar(ft.Container):
    def __init__(self, page: ft.Page, changed_rail_selected_index):
        global config
        super().__init__()
        self.PADDING = SIDEBAR_HALF_PADDING
        self.ICON_SIZE = ICON_SIZE
        self.TEXT = {
            "HOME": "ホーム",
            "STRUCTURE": "階層表示",
            "ROOT_FOLDER": "参照フォルダ",
            "BOOKMARK": "ブックマーク",
            "ALBUM": "アルバム",
            "PERSONAL_SETTINGS": "個人設定",
        }
        self.INDEX = {
            "HOME": 0,
            #"STRUCTURE": 4,
            "ROOT_FOLDER": 1,
            "BOOKMARK": 2,
            "ALBUM": 3,
            "PERSONAL_SETTINGS": 4,
        }

        self.page = page
        self.margin = ft.margin.only(top=TOOLBAR_HEIGHT)

        self.home = ft.NavigationRailDestination(
            icon_content=ft.Icon(ft.icons.HOME, size=self.ICON_SIZE),
            selected_icon_content=ft.Icon(ft.icons.HOME_OUTLINED),
            label_content=ft.Text(self.TEXT["HOME"]),
            padding=self.PADDING,
        )

        self.structure = ft.NavigationRailDestination(
            icon_content=ft.Icon(ft.icons.ACCOUNT_TREE, size=self.ICON_SIZE),
            selected_icon_content=ft.Icon(ft.icons.ACCOUNT_TREE_OUTLINED),
            label_content=ft.Text(self.TEXT["STRUCTURE"]),
            padding=self.PADDING,
        )

        self.folder = ft.NavigationRailDestination(
            icon_content=ft.Icon(ft.icons.FOLDER, size=self.ICON_SIZE),
            selected_icon_content=ft.Icon(ft.icons.FOLDER_OPEN),
            label_content=ft.Text(self.TEXT["ROOT_FOLDER"]),
            padding=self.PADDING,
        )

        self.personal_settings = ft.NavigationRailDestination(
            icon_content=ft.Icon(ft.icons.PERSON, size=self.ICON_SIZE),
            selected_icon_content=ft.Icon(ft.icons.PERSON_OUTLINE),
            label_content=ft.Text(self.TEXT["PERSONAL_SETTINGS"]),
            padding=self.PADDING,
        )

        self.bookmark =  ft.NavigationRailDestination(
            icon_content=ft.Icon(ft.icons.BOOKMARK, size=self.ICON_SIZE),
            selected_icon_content=ft.Icon(ft.icons.BOOKMARK_BORDER),
            label_content=ft.Text(self.TEXT["BOOKMARK"]),
            padding=self.PADDING,
        )

        self.album = ft.NavigationRailDestination(
            icon_content=ft.Icon(ft.icons.PHOTO_ALBUM, size=self.ICON_SIZE),
            selected_icon_content=ft.Icon(ft.icons.PHOTO_ALBUM_OUTLINED),
            label_content=ft.Text(self.TEXT["ALBUM"]),
            padding=self.PADDING,
        )

        self.destinations = [
            self.home,
            #self.structure,
            self.folder,
            self.bookmark,
            self.album,
            self.personal_settings,
        ]

        self.rail = ft.NavigationRail(
                selected_index=0,
                label_type=ft.NavigationRailLabelType.NONE,
                extended=False,
                min_width=SIDEBAR_MIN_WIDTH,
                min_extended_width=150,
                group_alignment=-1.0,
                destinations=self.destinations,
                on_change=changed_rail_selected_index,
                height=self.page.window_height,
            )
        
        self.last_selected_index = self.rail.selected_index

        self.content = self.rail


class ColorRail(ft.Column):
    def __init__(self, page: ft.Page):
        global config
        super().__init__()
        self.TEXT = {
            "LIGHT_THEME": "ライトテーマ",
            "DARK_THEME": "ダークテーマ",
            "COLOR_THEME": "カラーテーマ",
        }
        self.COLORS = [
            "RED", "PINK", "PURPLE", "INDIGO", "BLUE",
            "CYAN", "TEAL", "GREEN", "LIME", "YELLOW", "AMBER", 
            "ORANGE", "BROWN", "GREY", "WHITE", "BLACK",
        ]
        self.ICON_SIZE = ICON_SIZE

        self.page = page
        self.alignment = ft.MainAxisAlignment.END

        self.is_dark_mode = config["is_dark_mode"]

        if(self.is_dark_mode):
            theme_icon = ft.icons.LIGHT_MODE
            tooltip = self.TEXT["LIGHT_THEME"]
            self.page.theme_mode = ft.ThemeMode.DARK
        else:
            theme_icon = ft.icons.DARK_MODE
            tooltip = self.TEXT["DARK_THEME"]
            self.page.theme_mode = ft.ThemeMode.LIGHT

        self.color_mode_icon_button = ft.IconButton(
            icon=theme_icon,
            icon_size=self.ICON_SIZE,
            tooltip=tooltip,
            on_click=self.change_mode,
        )

        self.color_theme = config["color_theme"]
        self.page.theme = ft.theme.Theme(color_scheme_seed=self.color_theme)

        self.color_theme_icon_button = ft.IconButton(
            icon=ft.icons.PALETTE,
            icon_size=self.ICON_SIZE,
            icon_color=self.color_theme,
            tooltip=self.TEXT["COLOR_THEME"],
            on_click=self.change_theme,
        )

        self.controls = [
            ft.Container(self.color_theme_icon_button, padding=ft.padding.only(left=4*SIDEBAR_HALF_PADDING, right=4*SIDEBAR_HALF_PADDING)),
            ft.Container(self.color_mode_icon_button, padding=ft.padding.only(left=4*SIDEBAR_HALF_PADDING, right=4*SIDEBAR_HALF_PADDING, bottom=4*SIDEBAR_HALF_PADDING)),
        ]

    def change_mode(self, e):
        self.is_dark_mode = not self.is_dark_mode
        if(self.is_dark_mode):
            self.color_mode_icon_button.icon = ft.icons.LIGHT_MODE
            self.color_mode_icon_button.tooltip = self.TEXT["LIGHT_THEME"]
            self.page.theme_mode = ft.ThemeMode.DARK
        else:
            self.color_mode_icon_button.icon = ft.icons.DARK_MODE 
            self.color_mode_icon_button.tooltip = self.TEXT["DARK_THEME"]
            self.page.theme_mode = ft.ThemeMode.LIGHT
    
        config["is_dark_mode"] = self.is_dark_mode
        save_config()
        self.page.update()

    def change_theme(self, e):
        i = self.COLORS.index(self.color_theme)
        i += 1
        if(i >= len(self.COLORS)):
            i = 0
        self.color_theme = self.COLORS[i]
        self.color_theme_icon_button.icon_color = self.color_theme
        self.page.theme = ft.theme.Theme(color_scheme_seed=self.color_theme)
        config["color_theme"] = self.color_theme
        save_config()
        self.page.update()


class Gallery(ft.GridView):
    def __init__(self, page, open_image_func, image_type):
        super().__init__()
        self.padding = ft.padding.only(right=12, bottom=12, left=SIDEBAR_MIN_WIDTH + 2 * SIDEBAR_HALF_PADDING)
        self.spacing = 1
        self.run_spacing = 2
        self.expand = 1
        self.runs_count = 5
        self.horizontal = False
        self.child_aspect_ratio = 1.0
        self.image_type = image_type

        self.page = page
        self.on_scroll = self.changed_scrollbar
        self.page.on_window_event = self.on_focus

        self.path_list = []
        self.controls = []
        self.scroll_pixel = 0

        self.open_image = open_image_func

        self.error_image_content = ft.Container(
            content=ft.Icon(name=ft.icons.HIDE_IMAGE),
            alignment=ft.alignment.center,
            expand=True,
        )

        self.set_config()      

    def set_config(self):
        max_extent = 300
        image_fit = ft.ImageFit.CONTAIN

        if(self.image_type == "HOME"):
            max_extent = config["home_size"]
            image_fit = config["home_fit"]
        elif(self.image_type == "BOOKMARK"):
            max_extent = config["bookmark_size"]
            image_fit = config["bookmark_fit"]
        elif(self.image_type == "HISTORY"):
            max_extent = config["history_size"]
            image_fit = config["history_fit"]
        
        self.max_extent = max_extent
        self.image_fit = image_fit

    def make_image_container(self, image_path, album_title=None, specific_paths=None):
        def on_click_image(e):
            self.open_image(image_path, self.image_type, album_title, specific_paths)
 
        return ft.Container(
            ft.Image(
                src=image_path,
                fit=self.image_fit,
                error_content=self.error_image_content,
            ),
            on_click=on_click_image,
        )

    def set_pictures(self, album_title=None, search_paths=None):
        global config, paths, history, marks
        if(self.image_type == "HISTORY"):
            self.path_list = history["IMAGE"]
        elif(self.image_type == "BOOKMARK"):
            self.path_list = marks["BOOKMARK"]
        elif(self.image_type == "ALBUM"):
            self.path_list = marks["ALBUM"][album_title]
        elif(self.image_type == "SEARCH"):
            self.path_list = search_paths
        else:
            self.path_list = paths[self.image_type]
        self.images = [self.make_image_container(path, album_title, search_paths) for path in self.path_list]
        self.controls = self.images[:config["max_displayed_image"]]

    def add_pictures(self, reverse):
        global config, paths, history
        if(self.image_type == "HISTORY"):
            self.path_list = history["IMAGE"]
        else:
            self.path_list = paths[self.image_type]

        if(reverse):
            self.controls.reverse()

        self.images = [self.make_image_container(path) for path in self.path_list[len(self.controls):]]
        self.controls += self.images
        self.controls = self.controls[:config["max_displayed_image"]]

        if(reverse):
            self.controls.reverse()

    def changed_scrollbar(self, e: ft.OnScrollEvent):
        self.scroll_pixel = e.pixels

    def back_scroll(self):
        self.scroll_to(offset=self.scroll_pixel)
    
    def on_focus(self, e):
        try:
            if(e.data == "focus"):
                self.back_scroll()
        except:
            pass
        

class AlbumGalleries(ft.GridView):
    def __init__(self, page, open_album, open_image_func):
        super().__init__()
        self.MAX_ALBUM_BUM = 1000

        self.image_type = "ALBUM"
        self.padding = ft.padding.only(right=12, bottom=12, left=SIDEBAR_MIN_WIDTH + 2 * SIDEBAR_HALF_PADDING)
        self.spacing = 1
        self.run_spacing = 2
        self.expand = 1
        self.runs_count = 2
        self.max_extent = 300
        self.horizontal = False
        self.child_aspect_ratio = 1.0

        self.page = page
        self.open_album = open_album
        self.open_image_func = open_image_func

        self.path_list = []
        self.controls = []
        self.scroll_pixel = 0
        self.set_albums()

    def make_album_container(self, album_title):
        global marks
 
        return ft.Container(
            content=ft.Text(album_title, color=ft.colors.ON_PRIMARY_CONTAINER),
            on_click=lambda e, album_title=album_title: self.on_click_container(e, album_title),
            bgcolor=ft.colors.PRIMARY_CONTAINER
        )
    
    def set_albums(self):
        self.images = [self.make_album_container(title) for title in marks["ALBUM"].keys()]
        self.controls = self.images[:self.MAX_ALBUM_BUM]
        self.make_albums()

    def make_albums(self):
        global marks
        self.albums = {}
        for album_title in marks["ALBUM"].keys():
            self.albums[album_title] = Gallery(self.page, self.open_image_func, "ALBUM")

    def set_albums_pictures(self):
        for album_title in marks["ALBUM"].keys():
            self.albums[album_title].set_pictures(album_title=album_title)

    def on_click_container(self, e, album_title):        
        self.open_album(album_title)
        self.page.update()


class AlbumGalleryTab(ft.Container):
    def __init__(self, page, tab_index, close_func, open_browsing_history, album_gallery):
        super().__init__()
        self.TEXT = {
            "FEATURE_OF_IMAGE": "履歴内の画像の特徴",
            "DELETE_HISTORY": "閲覧履歴の削除",
            "DELETE_THEM_ALL?": "閲覧履歴をすべて削除しますか？",
            "YES": "はい",
            "CANCEL": "キャンセル",
        }
        self.ICON_SIZE = 24
        self.ICON_COLOR = ft.colors.ON_BACKGROUND
        self.BGCOLOR = ft.colors.BACKGROUND

        self.page = page
        self.tab_index = tab_index
        self.close_func = close_func
        self.open_browsing_history = open_browsing_history

        open_image = album_gallery.open_image
        image_type = album_gallery.image_type
        self.album_gallery = Gallery(self.page, open_image, image_type)
        self.album_gallery.controls = copy.copy(album_gallery.controls)

        self.toolbar = ToolBar("HISTORY", self.page, None, self.search_text_submit, self.close_tab, self.tab_index, None, self.open_browsing_history)
        self.toolbar.search_text.label = self.TEXT["FEATURE_OF_IMAGE"]

        self.set_content()

    def set_content(self):
        self.content = ft.Column(
            [
                self.toolbar,
                self.album_gallery
            ]
        )

    def search_text_submit(self, e):
        print(e)

    def close_tab(self, e):
        self.close_func(self.tab_index)


class FolderPicker(ft.Container):
    def __init__(self, page, set_pictures_func):
        super().__init__()
        global paths, logined
        self.TEXT = {
            "FOLDER_PICKING": "参照フォルダを選択",
            "PICK_IT!": "フォルダを選択してください。",
            "REFFERENCE": "参照: ",
            "PICKING_CANCELED": "フォルダの選択がキャンセルされました。",
            "GET_IMAGE_VECTORS": "画像ベクトルの抽出",
            "GET_IMAGE_V_EXPLANATION": "画像の検索ができるように画像から画像ベクトルを抽出します。\nこれには時間がかかる場合があります。",
            "COMPLETE_GETTING_VECTORS": "参照フォルダ内の画像に対して、画像ベクトルの抽出が完了しました。",
            "CONFIRM": "決定",
            "SHOULD_BE_OVER_256": "画像ベクトルを抽出する場合、画像が256枚以上である必要があります。",
            "OK": "確認",
            "CANCEL": "キャンセル",
        }
        self.set_pictures_func = set_pictures_func
        self.content_height = 350
        self.content_width = 250

        self.padding = ft.padding.only(top=TOOLBAR_HEIGHT, left=SIDEBAR_MIN_WIDTH + 2 * SIDEBAR_HALF_PADDING)
        self.page = page
        self.visible = False

        self.folder_is_selected = False

        pick_folder_dialog = ft.FilePicker(on_result=self.pick_folder_result)
        self.page.overlay.append(pick_folder_dialog)

        self.pick_button = ft.ElevatedButton(
            self.TEXT["FOLDER_PICKING"],
            icon="CREATE_NEW_FOLDER_OUTLINED",
            on_click=lambda _:pick_folder_dialog.get_directory_path()
        )

        self.get_vectors_button = ft.ElevatedButton(
            self.TEXT["GET_IMAGE_VECTORS"],
            icon=ft.icons.IMAGE_SEARCH,
            on_click=self.open_get_v_dialog,
            disabled=True,
        )
        self.get_vectors_button.disabled = not logined

        self.progressbar = ft.ProgressBar(width=250, height=20, value=0)
        self.progressbar2 = ft.ProgressBar(width=250, height=2, value=1)
        self.progressbar3 = ft.ProgressBar(width=250, height=2, value=1)

        self.folder_name = ft.Text()
        self.folder_size = ft.Text()
        self.folder_length = ft.Text()
        self.vec_length = ft.Text()
        self.time_text = ft.Text("")

        if("root_directory" in paths):
            self.folder_is_selected = True
        else:
            self.folder_is_selected = False

        if(not self.folder_is_selected):
            self.folder_name.value = self.TEXT["PICK_IT!"]
            self.folder_size.value = ""
            self.folder_length.value = ""
            self.vec_length.value = ""
        else:
            self.folder_name.value = self.TEXT["REFFERENCE"] + paths["root_directory"]
            self.folder_size.value = ""
            self.folder_length.value = ""
            self.vec_length.value = ""

        items = ft.Column(
            [
                ft.Container(self.pick_button, alignment=ft.alignment.center),
                self.folder_name,
                self.folder_length,
                self.folder_size,
                self.vec_length,
                ft.Container(
                    ft.Column(
                        [
                            self.progressbar3,
                            self.progressbar,
                            self.progressbar2,
                        ],
                        spacing=0
                    ),
                    margin=ft.margin.only(top=20, bottom=20),
                ),
                self.time_text,
                ft.Container(self.get_vectors_button, alignment=ft.alignment.center),
            ],
            alignment=ft.alignment.center
        )

        content = ft.Container(items, expand=True, width=self.content_width, height=self.content_height)
        self.alignment =ft.alignment.center
        self.expand=True
        self.content = content

        self.get_v_dialog = ft.AlertDialog(
            title=ft.Text(self.TEXT["GET_IMAGE_VECTORS"]),
            content=ft.Text(self.TEXT["GET_IMAGE_V_EXPLANATION"]),
            actions=[
                ft.TextButton(self.TEXT["CONFIRM"], on_click=self.open_over256_images_dialog),
                ft.TextButton(self.TEXT["CANCEL"], on_click=self.close_get_v_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.over256_images_dialog = ft.AlertDialog(
            content=ft.Text(self.TEXT["SHOULD_BE_OVER_256"]),
            actions=[
                ft.TextButton(self.TEXT["OK"], on_click=self.close_over256_images_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )


        self.complete_get_v = ft.SnackBar(ft.Text(self.TEXT["COMPLETE_GETTING_VECTORS"]))

    def load_folder(self):
        global loading
        global config, paths
        loading = True
        self.folder_name.value = self.TEXT["REFFERENCE"] + paths["root_directory"]
        self.progressbar.value = 0
        self.progressbar2.value = None
        self.progressbar3.value = None
        self.pick_button.disabled = True
        self.get_vectors_button.disabled = True
        self.page.update()
        time.sleep(1.0)
        get_picture_path(paths["root_directory"], config["image_formats"], self.progressbar)
        self.progressbar.value = 1
        self.page.update()
        self.set_pictures_func(initialize=True)
        self.page.update()

        self.load_folder_info()
        self.progressbar2.value = 1
        self.progressbar3.value = 1
        self.get_vectors_button.disabled = False
        loading = False
        self.page.update()

    def load_folder_info(self):
        global config, logined
        self.folder_length.value = f"画像ファイルの総数: {self.get_folder_length(paths['root_directory'], config['image_formats'])} 個"
        self.folder_size.value = f"画像ファイルの総データサイズ: {attach_a_unit(self.get_folder_size(paths['root_directory'], config['image_formats']))}"
        self.vec_length.value = f"読み込まれている画像ベクトル: {len(vectors['embeddings'])} 個"
        self.get_vectors_button.disabled = not logined
        self.pick_button.disabled = False
        self.page.update()

    def pick_folder_result(self, e):
        global paths
        if e.path:
            paths["root_directory"] = e.path
            save_paths()
            self.folder_name.value = "お待ちください"
            self.folder_size.value = ""
            self.folder_length.value = "読み込み中"
            self.vec_length.value = ""
            self.page.update()
            self.load_folder()
        else:
            self.page.snack_bar = ft.SnackBar(ft.Text(self.TEXT["PICKING_CANCELED"]))
            self.page.snack_bar.open = True
            self.page.update()

    def get_folder_size(self, path, image_formats):
        total_size = 0
        with os.scandir(path) as it:
            for entry in it:
                if(entry.is_file()):
                    if(check_format(entry.name, image_formats)):
                        total_size += entry.stat().st_size
                elif(entry.is_dir()):
                    total_size += self.get_folder_size(entry.path, image_formats)
        return total_size

    def get_folder_length(self, path, image_formats):
        total_size = 0
        with os.scandir(path) as it:
            for entry in it:
                if(entry.is_file()):
                    if(check_format(entry.name, image_formats)):
                        total_size += 1
                elif(entry.is_dir()):
                    total_size += self.get_folder_length(entry.path, image_formats)
        return total_size

    def open_get_v_dialog(self, e):
        self.page.dialog = self.get_v_dialog
        self.get_v_dialog.open = True
        self.page.update()

    def close_get_v_dialog(self, e):
        self.get_v_dialog.open = False
        self.page.update()

    def open_over256_images_dialog(self, e):
        self.close_get_v_dialog(e)
        if(len(paths["HOME"]) >= 256):
            self.get_vectors(e) 
        else:
            self.page.dialog = self.over256_images_dialog
            self.over256_images_dialog.open = True
            self.page.update()
            
    def close_over256_images_dialog(self, e):
        self.over256_images_dialog.open = False
        self.page.update()

    def get_vectors(self, e):
        global vectors, loading
        self.get_v_dialog.open = False
        loading = True
        self.progressbar.value = 0
        self.progressbar2.value = None
        self.progressbar3.value = None
        self.pick_button.disabled = True
        self.get_vectors_button.disabled = True
        
        self.page.update()

        get_model()
        self.image2vector()
        self.make_faiss_index()
    
        self.progressbar.value = 1
        self.progressbar2.value = 1
        self.progressbar3.value = 1
        self.pick_button.disabled = False
        self.time_text.value = ""
        loading = False
        self.load_folder_info()
        self.page.update()

    def image2vector(self):
        global paths, vectors, device
        global model, processor
        image_features = torch.Tensor().to(device)
        loaded_paths = []
        length = len(paths["HOME"])
        unit = 1.0 / length
        
        all_s = time.time()
        i = 0
        second_until_end = 0
        minute_until_end = 0
        with torch.no_grad():
            for image_path in paths["HOME"]:
                image = PIL.Image.open(image_path)
                processed_image = processor(images=image, return_tensors="pt").to(device)
                embedding = model.get_image_features(**processed_image)

                image_features = torch.cat((image_features, embedding), dim=0)
                self.progressbar.value += unit
                
                i += 1
                
                if(i % 50 == 0):
                    one_e = time.time()
                    time_until_end = ((one_e - all_s) / i) * length - (one_e - all_s)
                    second_until_end = round(time_until_end % 60)
                    minute_until_end = round(time_until_end / 60)
                    self.time_text.value = f"終了まで: およそ {minute_until_end}分 {second_until_end}秒"

                    self.progressbar.update()
                    self.time_text.update()
        
        self.progressbar.update()
        self.time_text.update()

        image_features = image_features.to('cpu').detach().numpy().copy()
        faiss.normalize_L2(image_features)
        loaded_paths = copy.deepcopy(paths["HOME"])
        vectors = {"paths": loaded_paths, "embeddings": image_features}
        save_vectors()
        self.open_complete_get_v_dialog()

    def make_faiss_index(self):
        global vectors
        dim = 768
        nlist = len(vectors["paths"]) // 50
        m = dim // 32
        nbits = 8

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

        index.train(vectors["embeddings"])
        index.add(vectors["embeddings"])

        vectors["index"] = index
        save_vectors()

    def open_complete_get_v_dialog(self):
        self.page.snack_bar = self.complete_get_v
        self.page.snack_bar.open = True
        self.page.update()


class PersonalSettings(ft.Container):
    def __init__(self, page):
        super().__init__()
        self.TEXT = {
            "HUGGING_FACE_ACCESS_TOKEN": "HuggingFace のアクセストークン",
            "INVALID_TOKEN": "無効なアクセストークンです。ログインに失敗しました。",
            "LOGIN_SUCCESSED": "ログインに成功しました。",        
        }

        self.alignment = ft.alignment.center
        self.margin = ft.margin.only(left=SIDEBAR_MIN_WIDTH)
        self.expand = True

        self.huggingface_access_token_textfield = ft.TextField(
            prefix_icon=ft.icons.KEY,
            border_color=ft.colors.ON_BACKGROUND,
            label=self.TEXT["HUGGING_FACE_ACCESS_TOKEN"],
            password=True, 
            can_reveal_password=True,
            on_submit=self.on_submit_access_token
        )

        self.error_snackbar = ft.SnackBar(ft.Text(self.TEXT["INVALID_TOKEN"]))
        self.success_snackbar = ft.SnackBar(ft.Text(self.TEXT["LOGIN_SUCCESSED"]))

        items = ft.Column(
            [
                ft.Container(self.huggingface_access_token_textfield, alignment=ft.alignment.center),
            ]
        )
        content = ft.Container(items, expand=True, width=400, height=250)
        self.content = content

    def on_submit_access_token(self, e):
        global logined
        token = self.huggingface_access_token_textfield.value
        try:
            huggingface_hub.login(token)
            self.huggingface_access_token_textfield.disabled = True
            logined = True
            self.page.snack_bar = self.success_snackbar
            self.page.snack_bar.open = True
            self.page.update()
        except:
            self.page.snack_bar = self.error_snackbar
            self.page.snack_bar.open = True
            self.page.update()


class MainWindow(ft.Stack):
    def __init__(self, page: ft.Page):
        super().__init__()
        self.TEXT = {
            "SPECIFY_A_FOLDER": "フォルダが指定されていません。",
            "NO_BOOKMARK": "ブックマークに登録された画像はありません。",
            "HOME_DISPLAY_SETTINGS": "表示設定：ホーム",
            "BOOKMARK_DISPLAY_SETTINGS": "表示設定：ブックマーク",
        }

        self.expand = True
        self.page = page
        self.page.on_resize = self.on_resize
        self.page.window_title_bar_hidden = True
        self.page.window_title_bar_buttons_hidden = True

        self.black_curtain = ft.Container(visible=False, expand=True, bgcolor=ft.colors.BLACK)
        self.sidebar = SideBar(self.page, self.changed_rail_selected_index)
        self.toolbar = ToolBar("HOME", self.page, self.sidebar, self.search_text_submit, self.close_tab, 0, self.reset_all_gallery, self.open_browsing_history)

        side_content = ft.Row(
            [
                self.sidebar,
                ft.VerticalDivider(width=1),
            ],
            spacing=0,
        )

        side_bottom_content = ColorRail(self.page)

        top_space_container = ft.Container(height=TOOLBAR_HEIGHT)
        self.gallery = Gallery(self.page, self.open_image, image_type="HOME")
        self.bookmark_gallery = Gallery(self.page, self.open_image, image_type="BOOKMARK")
        self.history_gallery = Gallery(self.page, self.open_image, image_type="HISTORY")
        self.album_galleries = AlbumGalleries(self.page, self.open_album, self.open_image)
        self.tabbar = TabBarColumn(self.page, self.close_all_tab, self.black_curtain, self.set_gallery_visiblility)
        self.tabbar.reset_gallery_scroll = self.reset_gallery_scroll

        self.history_gallery.visible = True
        for album_title in self.album_galleries.albums.keys():
            self.album_galleries.albums[album_title].visible = True

        self.progressring_container = ft.Column(
            [
                top_space_container,
                ft.Container(
                    content=ft.ProgressRing(width=SIDEBAR_MIN_WIDTH, height=SIDEBAR_MIN_WIDTH, stroke_width=SIDEBAR_MIN_WIDTH/20),
                    margin=ft.margin.only(left=SIDEBAR_MIN_WIDTH),
                    alignment=ft.alignment.center,
                    expand=True,
                )
            ]
        )

        self.home_content = ft.Column([top_space_container, self.gallery])
        self.home_text = ft.Container(
            ft.Container(ft.Text(self.TEXT["SPECIFY_A_FOLDER"]), height=200),
            alignment=ft.alignment.center, 
            margin=ft.margin.only(top=TOOLBAR_HEIGHT, left=SIDEBAR_MIN_WIDTH + 2 * SIDEBAR_HALF_PADDING),
            expand=True,
        )
        self.home_text.visible = False

        self.bookmark_content = ft.Column([top_space_container, self.bookmark_gallery])
        self.bookmark_text = ft.Container(
            ft.Container(ft.Text(self.TEXT["NO_BOOKMARK"]),  height=200),
            alignment=ft.alignment.center, 
            margin=ft.margin.only(top=TOOLBAR_HEIGHT, left=SIDEBAR_MIN_WIDTH + 2 * SIDEBAR_HALF_PADDING),
            expand=True,
        )
        self.bookmark_gallery.visible = False
        self.bookmark_text.visible = False

        self.folder_picker_content = FolderPicker(self.page, self.set_pictures_of_all)

        self.album_content = ft.Column([top_space_container, self.album_galleries])
        self.album_galleries.visible = False

        self.personal_settings_content = PersonalSettings(self.page)
        self.personal_settings_content.visible = False

        main_content = ft.Stack(
            [
                self.home_content,
                self.home_text,
                self.bookmark_content,
                self.bookmark_text,
                self.folder_picker_content,
                self.album_content,
                self.personal_settings_content,
                self.progressring_container,
                side_content, 
                side_bottom_content,
                self.toolbar,
            ]
        )

        self.tabbar.make_new_tab("HOME", "", main_content)
        
        self.controls = [self.black_curtain, self.tabbar]
        self.margin = 0
        self.padding = 0
        self.spacing = 0
        
        self.page.update()

    def set_gallery_visiblility(self, tab_type):
        sidebar_index = self.sidebar.rail.selected_index
        self.gallery.visible = (tab_type == "HOME" and sidebar_index == self.sidebar.INDEX["HOME"])
        self.bookmark_gallery.visible = (tab_type == "HOME" and sidebar_index == self.sidebar.INDEX["BOOKMARK"])
        self.album_galleries.visible = (tab_type == "HOME" and sidebar_index == self.sidebar.INDEX["ALBUM"])

    def open_image(self, image_path, image_type, album_title=None, specific_paths=None):
        global history
        self.black_curtain.visible = True

        tab_index = len(self.tabbar.tabs)
        if(image_type == "HOME"):
            path_list = self.gallery.path_list
        elif(image_type == "BOOKMARK"):
            path_list = self.bookmark_gallery.path_list
        elif(image_type == "HISTORY"):
            path_list = self.history_gallery.path_list
            path_list.reverse()
        elif(image_type == "SEARCH"):
            path_list = specific_paths
        elif(image_type == "ALBUM"):
            path_list = self.album_galleries.albums[album_title].path_list
        tab = ImageTab(tab_index, self.close_tab, self.page, image_path, path_list, self.tabbar.change_tab_display, self.bookmark_gallery.set_pictures)
        self.tabbar.make_new_tab("IMAGE", image_path, tab)

        self.page.update()

    def close_tab(self, tab_index, deleted=False):
        self.black_curtain.visible = False

        if(deleted and self.tabbar):
            image_path = self.tabbar.tabbar_content[tab_index].path
            send2trash.send2trash(image_path)
            self.delete_image_in_all_gallery(image_path)
        
        self.tabbar.close_tab(tab_index)
        self.gallery.back_scroll()

    def delete_image_in_all_gallery(self, image_path):
        global config, paths, history, marks
        index = self.gallery.path_list.index(image_path)
        del self.gallery.path_list[index]
        del self.gallery.images[index]
        self.gallery.controls = self.gallery.images[:config["home_num"]]
        paths["HOME"] = self.gallery.path_list
        save_paths()

        try:
            index = self.bookmark_gallery.path_list.index(image_path)
            del self.bookmark_gallery.path_list[index]
            del self.bookmark_gallery.images[index]
            self.bookmark_gallery.controls = self.bookmark_gallery.images[:config["bookmark_num"]]
            marks["BOOKMARK"] = self.bookmark_gallery.path_list
            save_marks()
        except:
            pass

        try:
            index = self.history_gallery.path_list.index(image_path)
            del self.history_gallery.path_list[index]
            del self.history_gallery.images[index]
            self.history_gallery.controls = self.history_gallery.images[:config["history_num"]]
            history["IMAGE"] = self.history_gallery.path_list
            save_history()
        except:
            pass

    def search_text_submit(self, e):
        global loading
        query = e.control.value
        if((not loading) and query != ""):
            tab_index = len(self.tabbar.tabs)
            tab = SearchTab(tab_index, self.page, query, self.search_text_submit, self.close_tab, self.reset_all_gallery, self.open_browsing_history, self.open_image)
            self.tabbar.make_new_tab("SEARCH", query, tab)
            self.page.update()

            result_paths = self.vector_search(query)
            self.tabbar.tabbar_content[tab_index].search_gallery.set_pictures(search_paths=result_paths)
            self.page.update()

    def vector_search(self, query):
        global config, vectors
        global tokenizer, model
        text = tokenize(
            tokenizer=tokenizer,
            texts=[query],
        ).to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**text)
            text_features = text_features.to('cpu').detach().numpy().copy()
            faiss.normalize_L2(text_features)

        index = vectors["index"]
        D, I = index.search(text_features, config["top_n"])

        return [vectors["paths"][j] for j in I[0]]

    def open_album(self, album_title):
        tab_index = len(self.tabbar.tabs)
        already_opened = False
        already_opened_index = -1
        for i in range(tab_index):
            if(self.tabbar.tabs[i].tab_type == "ALBUM" and self.tabbar.tabbar.tabs[i].tab_content.controls[1].value == album_title):
                already_opened = True
                already_opened_index = i
                break
        if(not already_opened):
            tab = AlbumGalleryTab(self.page, tab_index, self.close_tab, self.open_browsing_history, self.album_galleries.albums[album_title])
            self.tabbar.make_new_tab("ALBUM", album_title, tab)
            self.album_galleries.visible = False
            self.album_galleries.albums[album_title].visible = True
        else:
            self.tabbar.tabbar.selected_index = already_opened_index
            self.tabbar.reset_visibility(already_opened_index)
        self.page.update()

    def open_browsing_history(self, e):
        tab_index = len(self.tabbar.tabs)

        already_opened = False
        already_opened_index = -1
        for i in range(tab_index):
            if(self.tabbar.tabs[i].tab_type == "HISTORY"):
                already_opened = True
                already_opened_index = i
                break
        if(not already_opened):
            tab = BrowsingHistoryTab(self.page, tab_index, self.close_tab, self.open_browsing_history, self.history_gallery)
            self.tabbar.make_new_tab("HISTORY", "history", tab)
            self.history_gallery.add_pictures(True)
        else:
            self.tabbar.tabbar.selected_index = already_opened_index
            self.tabbar.reset_visibility(already_opened_index)

        self.page.update()

    def close_all_tab(self, e):
        if(len(self.tabbar.tabs) >= 2):
            for i in range(1, len(self.tabbar.tabs)):
                self.tabbar.close_last_tab()
        self.page.update()

    def changed_rail_selected_index(self, e):
        if(not loading):
            i = e.control.selected_index

            self.gallery.visible = (i == self.sidebar.INDEX["HOME"])
            self.home_text.visible = (i == self.sidebar.INDEX["HOME"] and len(self.gallery.controls) <= 0)
            if(i == self.sidebar.INDEX["HOME"]):
                self.gallery.back_scroll()
                self.toolbar.tab_type = "HOME"
                self.toolbar.display_settings_popup.title = ft.Text(self.TEXT["HOME_DISPLAY_SETTINGS"])

            if(i == self.sidebar.INDEX["ROOT_FOLDER"]):
                self.toolbar.tab_type = "ROOT_FOLDER"
            self.folder_picker_content.visible = (i == self.sidebar.INDEX["ROOT_FOLDER"])
            self.folder_picker_content.get_vectors_button.disabled = not logined


            self.bookmark_gallery.visible = (i == self.sidebar.INDEX["BOOKMARK"])
            self.bookmark_text.visible = (i == self.sidebar.INDEX["BOOKMARK"] and len(self.bookmark_gallery.controls) <= 0)
            if(i == self.sidebar.INDEX["BOOKMARK"]):
                self.toolbar.tab_type = "BOOKMARK"
                self.toolbar.display_settings_popup.title = ft.Text(self.TEXT["BOOKMARK_DISPLAY_SETTINGS"])

            self.album_galleries.visible = (i == self.sidebar.INDEX["ALBUM"])
            #"ALBUM",
            if(i == self.sidebar.INDEX["ALBUM"]):
                self.toolbar.tab_type = "ALBUM"
            
            self.personal_settings_content.visible = (i == self.sidebar.INDEX["PERSONAL_SETTINGS"])

            self.toolbar.set_settings_value()
            self.page.update()

    def set_pictures_of_all(self, initialize):
        self.gallery.set_pictures()
        self.bookmark_gallery.set_pictures()
        self.history_gallery.set_pictures()
        self.album_galleries.set_albums_pictures()
        if(initialize):
            self.reset_all_gallery()
    
    def reset_all_gallery(self, change_dict=None):
        global paths, config, marks
        random.seed()

        if(change_dict):
            change_sorting = change_dict["sorting"]
            change_fit = change_dict["fit"]
            change_size = change_dict["size"]
            change_num = change_dict["num"]
        else:
            change_sorting = True
            change_fit = False
            change_size = False
            change_num = True
        
        if(change_sorting):
            if(self.toolbar.tab_type == "HOME" and len(self.gallery.path_list) > 0):
                if(config["home_sorting"] == self.toolbar.TEXT["RANDOM"]):
                    tmp = list(zip(self.gallery.path_list, self.gallery.images))
                    random.shuffle(tmp)
                    self.gallery.path_list, self.gallery.images = zip(*tmp)
                    self.gallery.path_list, self.gallery.images = list(self.gallery.path_list), list(self.gallery.images)
                elif(config["home_sorting"] == self.toolbar.TEXT["ASCENDING"]):
                    self.gallery.path_list, self.gallery.images = zip(*sorted(zip(self.gallery.path_list, self.gallery.images), key=lambda x: os.path.getctime(x[0])))
                    self.gallery.path_list, self.gallery.images = list(self.gallery.path_list), list(self.gallery.images)

                elif(config["home_sorting"] == self.toolbar.TEXT["DESCENDING"]):
                    self.gallery.path_list, self.gallery.images = zip(*sorted(zip(self.gallery.path_list, self.gallery.images), key=lambda x: os.path.getctime(x[0])))
                    self.gallery.path_list, self.gallery.images = list(self.gallery.path_list), list(self.gallery.images)
                    self.gallery.path_list.reverse()
                    self.gallery.images.reverse()
                self.gallery.controls = self.gallery.images[:config["max_displayed_image"]]

            elif(self.toolbar.tab_type == "BOOKMARK" and len(self.bookmark_gallery.path_list) > 0):
                if(config["bookmark_sorting"] == self.toolbar.TEXT["RANDOM"]):
                    tmp = list(zip(self.bookmark_gallery.path_list, self.bookmark_gallery.images))
                    random.shuffle(tmp)
                    self.bookmark_gallery.path_list, self.bookmark_gallery.images = zip(*tmp)
                elif(config["bookmark_sorting"] == self.toolbar.TEXT["ASCENDING"]):
                    self.bookmark_gallery.path_list, self.bookmark_gallery.images = zip(*sorted(zip(self.bookmark_gallery.path_list, self.bookmark_gallery.images), key=lambda x: os.path.getctime(x[0])))
                    self.bookmark_gallery.path_list, self.bookmark_gallery.images = list(self.bookmark_gallery.path_list), list(self.bookmark_gallery.images)
                elif(config["bookmark_sorting"] == self.toolbar.TEXT["DESCENDING"]):
                    self.bookmark_gallery.path_list, self.bookmark_gallery.images = zip(*sorted(zip(self.bookmark_gallery.path_list, self.bookmark_gallery.images), key=lambda x: os.path.getctime(x[0])))
                    self.bookmark_gallery.path_list, self.bookmark_gallery.images = list(self.bookmark_gallery.path_list), list(self.bookmark_gallery.images)
                    self.bookmark_gallery.path_list.reverse()
                    self.bookmark_gallery.images.reverse()
                self.bookmark_gallery.controls = self.bookmark_gallery.images[:config["max_displayed_image"]]

            paths["HOME"] = self.gallery.path_list
            marks["BOOKMARK"] = self.bookmark_gallery.path_list
            self.gallery.scroll_to(offset=0)
            self.bookmark_gallery.scroll_to(offset=0)
            save_paths()
            save_marks()

        if(change_fit):
            if(self.toolbar.tab_type == "HOME" and len(self.gallery.path_list) > 0):
                how_to_fit = config["home_fit"]
                for i in range(len(self.gallery.images)):
                    self.gallery.images[i].content.fit = how_to_fit
            elif(self.toolbar.tab_type == "BOOKMARK" and len(self.bookmark_gallery.path_list) > 0):
                how_to_fit = config["bookmark_fit"]
                for i in range(len(self.bookmark_gallery.images)):
                    self.bookmark_gallery.images[i].content.fit = how_to_fit
            elif(self.toolbar.tab_type == "HISTORY" and len(self.history_gallery.path_list) > 0):
                how_to_fit = config["history_fit"]
                for i in range(len(self.history_gallery.images)):
                    self.history_gallery.images[i].content.fit = how_to_fit

        if(change_size):
            if(self.toolbar.tab_type == "HOME" and len(self.gallery.path_list) > 0):
                self.gallery.max_extent = config["home_size"]
            elif(self.toolbar.tab_type == "BOOKMARK" and len(self.bookmark_gallery.path_list) > 0):
                self.bookmark_gallery.max_extent = config["bookmark_size"]
            elif(self.toolbar.tab_type == "HISTORY" and len(self.history_gallery.path_list) > 0):
                self.history_gallery.max_extent = config["history_size"]

        if(change_num):
            if(self.toolbar.tab_type == "HOME" and len(self.gallery.path_list) > 0):
                self.gallery.controls = self.gallery.images[:config["home_num"]] 
            elif(self.toolbar.tab_type == "BOOKMARK" and len(self.bookmark_gallery.path_list) > 0):
                self.bookmark_gallery.controls = self.bookmark_gallery.images[:config["bookmark_num"]] 
            elif(self.toolbar.tab_type == "HISTORY" and len(self.history_gallery.path_list) > 0):
                self.history_gallery.controls = self.history_gallery.images[:config["history_num"]] 

        self.page.update()

    def reset_gallery_scroll(self, tab_type):
        if(tab_type == "HOME"):
            self.gallery.back_scroll()
            self.bookmark_gallery.back_scroll()
        elif(tab_type == "HISTORY"):
            self.history_gallery.back_scroll()

    def on_resize(self, e):
        global config
        self.sidebar.height = self.page.window_height
        config["window_height"] = self.page.window_height
        config["window_width"] = self.page.window_width
        config["window_maximized"] = self.page.window_maximized
        save_config()

        for i in range(len(self.tabbar.tabs)):
            if(type(self.tabbar.tabbar_content[i]) == ImageTab):
                self.tabbar.tabbar_content[i].image.width = self.page.window_width
                self.tabbar.tabbar_content[i].image.height = self.page.window_height

        self.page.update()


class GYGES():
    def __init__(self):
        ft.app(target=self.main)

    def main(self, page: ft.Page):
        global paths
        self.page = page

        self.get_source()
        self.set_layout()

        self.main_window = MainWindow(self.page)
        self.page.add(self.main_window)
        self.page.update()

        if("root_directory" in paths):
            self.main_window.set_pictures_of_all(initialize=False)
            self.main_window.progressring_container.visible = False
        else:
            self.main_window.progressring_container.visible = False
            self.main_window.home_text.visible = True

        self.page.update()
        
        if("root_directory" in paths):
            self.main_window.folder_picker_content.load_folder_info()
            self.page.update()

        try:
            get_model()
        except:
            pass
        
    def get_source(self):
        global config, paths, history, marks, vectors
        self.TEXT = {
            "RANDOM": "ランダム",
            "CONTAIN": "CONTAIN",
        }

        if(not os.path.isdir(CONFIG_DIR_NAME)):
            os.mkdir(CONFIG_DIR_NAME)

        if(not os.path.isdir(SOURCE_DIR_NAME)):
            os.mkdir(SOURCE_DIR_NAME)

        # config------------------------------------------------
        
        if(os.path.isfile(CONFIG_PATH)):
            with open(CONFIG_PATH, "rb") as f:
                config = pickle.load(f)
        

        config["image_formats"] = [".png", ".jpg", ".JPG", ".jpeg", ".jfif", ".gif", ".webp", ".bmp", ".wbmp"]

        if(not "is_dark_mode" in config):
            config["is_dark_mode"] = False
        if(not "color_theme" in config):
            config["color_theme"] = "BLUE"

        if(not "max_displayed_image" in config):
            config["max_displayed_image"] = 1000

        if(not "home_sorting" in config):
            config["home_sorting"] = self.TEXT["RANDOM"]
        if(not "bookmark_sorting" in config):
            config["bookmark_sorting"] = self.TEXT["RANDOM"]
        if(not "history_sorting" in config):
            config["history_sorting"] = self.TEXT["RANDOM"]

        if(not "home_fit" in config):
            config["home_fit"] = self.TEXT["CONTAIN"]
        if(not "bookmark_fit" in config):
            config["bookmark_fit"] = self.TEXT["CONTAIN"]
        if(not "history_fit" in config):
            config["history_fit"] = self.TEXT["CONTAIN"]

        if(not "home_size" in config):
            config["home_size"] = 300
        if(not "bookmark_size" in config):
            config["bookmark_size"] = 300
        if(not "history_size" in config):
            config["history_size"] = 300

        if(not "home_num" in config):
            config["home_num"] = 1000
        if(not "bookmark_num" in config):
            config["bookmark_num"] = 1000
        if(not "history_num" in config):
            config["history_num"] = 1000

        if(not "top_n" in config):
            config["top_n"] = 50

        # history---------------------------------------------
        if(os.path.isfile(HISTORY_PATH)):
            with open(HISTORY_PATH, "rb") as f:
                history = pickle.load(f)

        if(not "IMAGE" in history):
            history["IMAGE"] = []

        if(not "SEARCH" in history):
            history["SEARCH"] = []

        # path------------------------------------------------

        if(os.path.isfile(PATHES_PATH)):
            with open(PATHES_PATH, "rb") as f:
                paths = pickle.load(f)
            
        if(not "HOME" in paths):
            paths["HOME"] = []

        # marks------------------------------------------------

        if(os.path.isfile(MARKS_PATH)):
            with open(MARKS_PATH, "rb") as f:
                marks = pickle.load(f)
            
        if(not "BOOKMARK" in marks):
            marks["BOOKMARK"] = []
        if(not "ALBUM" in marks):
            marks["ALBUM"] = {}

        save_marks()

        # vector------------------------------------------------

        if(os.path.isfile(VECTRORS_PATH)):
            with open(VECTRORS_PATH, "rb") as f:
                vectors = pickle.load(f)

        if(not "paths" in vectors):
            vectors["paths"] = []
        if(not "embeddings" in vectors):
            vectors["embeddings"] = []
        if(not "index" in vectors):
            vectors["index"] = []

    def set_layout(self):
        global config
        self.page.title = "Gyges"
        self.page.padding = 0
        self.page.theme = ft.theme.Theme(color_scheme_seed="GREY")

        if("window_height" in config):
            self.page.window_height = config["window_height"]
        if("window_width" in config):
            self.page.window_width = config["window_width"]
        if("window_maximized" in config):
            self.page.window_maximized = config["window_maximized"]

        self.page.window_left = 0
        self.page.window_top = 0

    

if __name__ == "__main__":
    gyges = GYGES()
