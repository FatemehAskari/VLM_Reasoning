import os
from PIL import Image, ImageDraw, ImageFont


class GridDrawer:
    def draw(self, img, draw, rows, cols):
        raise NotImplementedError()


class FullGridDrawer(GridDrawer):
    def draw(self, img, draw, rows, cols):
        w, h = img.size
        for i in range(1, rows):
            y = h * i / rows
            draw.line([(0, y), (w, y)], fill=(0, 0, 0), width=1)
        for j in range(1, cols):
            x = w * j / cols
            draw.line([(x, 0), (x, h)], fill=(0, 0, 0), width=1)


class FullGridNumberedDrawer(FullGridDrawer):
    def __init__(self, font):
        self.font = font

    def draw(self, img, draw, rows, cols):
        super().draw(img, draw, rows, cols)
        w, h = img.size
        for i in range(rows):
            for j in range(cols):
                x = int(j * w / cols + w / (2 * cols))
                y = int(i * h / rows + h / (2 * rows))
                draw.text((x, y), f"{i + 1}-{j + 1}", fill=(0, 0, 0), font=self.font, anchor="mm")


class RowOnlyDrawer(GridDrawer):
    def draw(self, img, draw, rows, cols):
        w, h = img.size
        for i in range(1, rows):
            y = h * i / rows
            draw.line([(0, y), (w, y)], fill=(0, 0, 0), width=1)


class RowOnlyNumberedDrawer(RowOnlyDrawer):
    def __init__(self, font):
        self.font = font

    def draw(self, img, draw, rows, cols):
        super().draw(img, draw, rows, cols)
        w, h = img.size
        for i in range(rows):
            y = int(i * h / rows + h / (2 * rows))
            draw.text((5, y), f"{i + 1}", fill=(0, 0, 0), font=self.font, anchor="lm")


class ColumnOnlyDrawer(GridDrawer):
    def draw(self, img, draw, rows, cols):
        w, h = img.size
        for j in range(1, cols):
            x = w * j / cols
            draw.line([(x, 0), (x, h)], fill=(0, 0, 0), width=1)


class ColumnOnlyNumberedDrawer(ColumnOnlyDrawer):
    def __init__(self, font):
        self.font = font

    def draw(self, img, draw, rows, cols):
        super().draw(img, draw, rows, cols)
        w, h = img.size
        for j in range(cols):
            x = int(j * w / cols + w / (2 * cols))
            draw.text((x, 5), f"{j + 1}", fill=(0, 0, 0), font=self.font, anchor="ma")


class GradientDotsDrawer(GridDrawer):
    def __init__(self, dot_radius=4):
        self.dot_radius = dot_radius

    def draw(self, img, draw, rows, cols):
        w, h = img.size
        for i in range(rows + 1):
            y = i * h / rows
            for j in range(cols + 1):
                x = j * w / cols
                grad = (i + j) / (rows + cols)
                gray = int(255 * (1 - grad))
                draw.ellipse(
                    [(x - self.dot_radius, y - self.dot_radius), (x + self.dot_radius, y + self.dot_radius)],
                    fill=(gray, gray, gray),
                    outline=(0, 0, 0)
                )

class SequentialNumberedGridDrawer(FullGridDrawer):
    def __init__(self, font):
        self.font = font

    def draw(self, img, draw, rows, cols):
        super().draw(img, draw, rows, cols)
        w, h = img.size
        number = 1
        for i in range(rows):
            for j in range(cols):
                x = int(j * w / cols + w / (2 * cols))
                y = int(i * h / rows + h / (2 * rows))
                draw.text((x, y), str(number), fill=(0, 0, 0), font=self.font, anchor="mm")
                number += 1

class ImageGridProcessor:
    def __init__(self, drawer: GridDrawer):
        self.drawer = drawer

    def process_image(self, input_path, output_path, rows, cols):
        img = Image.open(input_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        self.drawer.draw(img, draw, rows, cols)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)


def load_font(size=18):
    paths = ["LiberationSerif-Regular.ttf"]
    for path in paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()