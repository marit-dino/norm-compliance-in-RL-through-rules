import tkinter as tk


class Interface(tk.Frame):

    def __init__(self, instance):
        self.frame = tk.Tk()
        self.instance = instance

        # size for each cell in pixel
        self.size = 32

        # colors for the board
        self.color1 = "white"
        self.color2 = "grey"
        self.pieces = {}

        canvas_width = self.instance.size * self.size
        canvas_height = self.instance.size * self.size

        tk.Frame.__init__(self, self.frame)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=canvas_width, height=canvas_height,
                                background="bisque")
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)
        # this binding will cause a refresh if the user interactively
        # changes the window size
        self.canvas.bind("<Configure>", self.draw)

        self.pack(side="top", fill="both", expand=True, padx=4, pady=4)
        self.player_image = tk.PhotoImage(data=IMAGEDATA_PLAYER)
        self.target_image = tk.PhotoImage(data=IMAGEDATA_TARGET)
        self.plant_image = tk.PhotoImage(data=IMAGEDATA_PLANT)
        self.frog_image = tk.PhotoImage(data=IMAGEDATA_FROG)

        self.add_piece("player", self.player_image, self.instance.player)
        self.add_piece("target", self.target_image, self.instance.target)
        for count, plant in enumerate(self.instance.plants):
            self.add_piece("plant_" + str(count), self.plant_image, plant)
        if hasattr(self.instance, "frogs"):
            for count, frog in enumerate(self.instance.frogs):
                self.add_piece("frog_" + str(count), self.frog_image, frog)

    def add_piece(self, name, image, position):
        '''Add a piece to the playing board'''
        self.canvas.create_image(0, 0, image=image, tags=(name, "piece"),
                                 anchor="c")
        self.place_piece(name, position)

    def place_piece(self, name, position):
        '''Place a piece at the given row/column'''
        self.pieces[name] = position
        x0 = ((position[0] - 1) * self.size) + int(self.size / 2)
        y0 = ((position[1] - 1) * self.size) + int(self.size / 2)
        self.canvas.coords(name, x0, y0)

    def remove_pieces(self, name):
        remove_keys = []
        for key in self.pieces.keys():
            if key.startswith(name):
                remove_keys.append(key)
        for key in remove_keys:
            self.pieces.pop(key)
            self.canvas.delete(key)

    def draw(self, event):
        '''Redraw the board, possibly in response to window being resized'''
        xsize = int(event.width / self.instance.size)
        ysize = int(event.height / self.instance.size)
        self.size = min(xsize, ysize)
        self.canvas.delete("square")
        for row in range(self.instance.size):
            for col in range(self.instance.size):
                color = self.color1 if (col + 1,
                                        row + 1) not in self.instance.walls else \
                    self.color2
                x1 = (col * self.size)
                y1 = (row * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black",
                                             fill=color, tags="square")
        for name in self.pieces:
            self.place_piece(name, self.pieces[name])
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")


IMAGEDATA_PLAYER = '''
    iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAAXNSR0IArs4c6QAAAERlWElmTU0
    AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAGKADAAQAAA
    ABAAAAGAAAAADiNXWtAAAAwklEQVRIDWNgGEnAE+jZx1AMYlMdgAz/D8UgNlGAiShVdFJEVhAxo
    jluBpCvB8QngfgGEN8B4hdA/BqIvwLxTyAGAXYg5gZiUSCWAGIVINYAYnMgvgTEGUCMFdwHisLC
    mVwaZAYcoPtADihjDcSGQKwJxMpALAPEvECMDXwGCj4B4rtAfB2IzwPxUSB+BMRggG4BTBydBvk
    GGyCon+apaNQCbPGCIjbyggiULAkmTeQwGnlBhOx7othDP4iI8uagVgQAnP4rQpmZ8u4AAAAASU
    VORK5CYII=
    '''

IMAGEDATA_PLANT = '''
    iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAAXNSR0IArs4c6QAAAERlWElmTU0
    AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAGKADAAQAAA
    ABAAAAGAAAAADiNXWtAAABOUlEQVRIDe2US0oDURBFoyC6BN2BQTQoRLPKoPE30J04EMkW1IkzI
    TjXgR8ieg6kQtHE0Hb3sC8cuvJevfq97nQ6rf45gR/85RXuYQhdaEyRID+nRD+BtSayvBHE4Fsw
    gAv4AtduYB1q6ZHTBjtMUfrYL7P107SuuQPH8ABRnOf/1IgdHTyUZRI7+YZtsJNzcHz6F2FpsXo
    sG+Qd9goujstAl3A3sz95noHjdKzu28lSXbGr4zPkJAbJlU74fQCh2HdcS7XB7hgMZieOyzuJCl
    3/gH3IshP3iuPNPnPbJNfguHLVYXtXWXbiuLwTL760rNLZP4FVR4Ij7JA+jss9L76yVjgZCTaxT
    WInkfgWu/Z3Egny0zH6ZtUOTox5B76Kvi3+hexCY4rKSwdcLe1Z0bFNUHFw7bE0gV8HXmSOKfy9
    ngAAAABJRU5ErkJggg==
    '''

IMAGEDATA_TARGET = '''
    iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAAXNSR0IArs4c6QAAAERlWElmTU0
    AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAGKADAAQAAA
    ABAAAAGAAAAADiNXWtAAABb0lEQVRIDd3UMU4DQQwF0ECkVHQ0CPpcgoKalgJqGgR34Ry5AiUHo
    KFGdCDlABSU8F8YS4ElG4SyTb70d2bsb3t2PLuj0bZjkhc8D2fhU/jWaM7GR/MvnCXqOfxYQxra
    P2Mc5W1YiR8yvw6n4V6jORtf6cSIXYtK/h7lZbjTE8FHQ6uQ2F54VUIBx025m/EmvAvnjeZsfEB
    bRVYel2bVmdsVHIX3YR3Dz5GPBsTwy/Fr490IAufq1e1OAraX8CI8aDRn46OhFVM9kauDWSwCNA
    8cQSXfX1i+P9iqCC2IFSNXB+4257R5nLO13cJpKCGaAx8NLYi1lqsDHxGnqwgaau1YoHbLZg581
    rQg1lquBeoW1Hrj43KB2sVhq/LYxpM2XmV8bTSH8pW2YivXl6o9Zxm93mBNHvyaTrL7QT80JzXo
    r0IBGPRnp8B4qYimb/x3rQg4ruqJQqtIs/IPGl8vNN7tcoV9/r5QNGfjo9lifALEXqkt2d8oHAA
    AAABJRU5ErkJggg==
    '''

IMAGEDATA_FROG = '''
    iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAACXBIWXMAAAsTAAALEwEAmpwYAAA
    CAUlEQVR4nO3VS4hOYRgH8N+4FzILcqeUMkWmSTRWLhtlSE0pKSVKuaRILqVsRJFLEVJ2mFiwkF
    sky9HkkizIdcFiMMY1IfTq+XKavr5zGHbfv9465znPef/vc/u/VFHFb/TCUmzEcP8Zo9GCeziHh
    2jGJNT8S6J+2IOv+BHEB9CJZ3iPJ1j5p8R9MBQj0DNsE3ETbajHU2zGKnxAA3pHpI9xqAjRdFzA
    x4ggre/owCfsQt/wnRYHSJGtzewxDJdxqRJRD+yNjVMT1EWUaY1BY0RbCTVYHnucRG0l59243Y1
    uG4DrkeamPOdZeB0N8LeYF3Xrr0AaUmSbdA87cLyI42K0R0q6gytYnee0MOamlPPTuIPtmJpTgj
    QyJdRGV0+uRNYQszMnYxuERTiBt7iG8WW6+QVmZmzrcDcvuiRH+yp8H4KjeIUJGXt9HGYUjmEc3
    mB+HmHSwLl5TjiMs5n3BbgVc9kZMra/wD6/5q6ty+nLoTnqWsIUvItU38epopqZ5OlIFPsGtnX5
    Pji6LnXwhpjRpojmM1qxJjJV0tpcJKE9Hyc+mCF6EPqZ1dLSc9LZGUF4JoZ9WVHCdOpvuIgtcaU
    sietmZ4zH+ujcNNRfsCL+HYnnIdztYU86OhsDK5E2xuYtQdwatXkUq6PLeomx8W9dxu9q3BAp4q
    1FI66iCuXwE3+he3aGwAhlAAAAAElFTkSuQmCC
    '''
