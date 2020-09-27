"""Noise Layers are for now on just layers like the 'normal' layers."""

from random import randint, choice, random
from PIL import ImageDraw, ImageFilter, ImageEnhance

from .layers import Layer


class Noise(Layer):
    """Root Object of Noise.
    By default, identity layer.
    """

    def __init__(self, name='Noise'):
        """
        Arguments:
            name: A string,
                the name of the noise layer.

        """

        if name is None:
            raise ValueError('')
        self.name = name

    def call(self, img):
        if img is None:
            raise ValueError('img is None')
        return img

    def summary(self):
        return self.name


class Shadows(Noise):
    """Adds shadows to the image."""

    # TODO: add transparency in the shadows

    def __init__(self, color=(255, 255, 255), name='Shadows'):
        """
        Arguments:
            color: A tuple of length 3,
                the RGB values for the shadow.
            name: A string,
                the name of the Layer.
        """

        if name is None:
            name = 'Shadows'
        if color is None:
            color = (255, 255, 255)

        super(Shadows, self).__init__()

        self.color = color
        self.name = name

    def call(self, img):

        if img is None:
            raise ValueError('img is None')

        x1 = randint(0, img.width)
        x2 = randint(0, img.width)
        y1 = randint(0, img.height)
        y2 = 10000000
        c = choice(self.color.colors)

        while abs(y2 - y1) > 75:
            if randint(0, 1):
                y2 = randint(y1, img.height)
            else:
                y2 = randint(0, y1)

        draw = ImageDraw.Draw(img)
        draw.rectangle((x1, y1, x2, y2), fill=c, outline=c)
        del draw

        return img


class Filter(Noise):
    """Adds filters to the image."""

    def __init__(self, blur=0, gauss_blur=0, smooth=0, smooth_more=0, rank_filter=0, name='Filter'):
        """
        Arguments:
            blur:

            gauss_blur:

            smooth:

            smooth_more:

            rank_filter:

            name: A string,
                the name of the Layer.
        """

        if name is None:
            raise ValueError
        if not all([item is not None for item in [blur, gauss_blur, smooth, smooth_more, rank_filter]]):
            raise ValueError
        if blur + gauss_blur + smooth + smooth_more + rank_filter > 1:
            raise ValueError
        if not all(0 <= item <= 1 for item in [blur, gauss_blur, smooth, smooth_more, rank_filter]):
            raise ValueError

        super(Filter, self).__init__()

        self.blur = blur
        self.gauss_blur = gauss_blur
        self.smooth = smooth
        self.smooth_more = smooth_more
        self.rank_filter = rank_filter

        self.name = name

    def call(self, img):

        if img is None: raise ValueError('img is None')

        im_n = img.copy()

        gauss_blur_low, gauss_blur_high = 0, self.gauss_blur
        blur_low, blur_high = gauss_blur_high, gauss_blur_high + self.blur
        smooth_low, smooth_high = blur_high, blur_high + self.smooth
        smooth_more_low, smooth_more_high = smooth_high, smooth_high + self.smooth_more
        rank_low, rank_high = smooth_more_high, smooth_more_high + self.rank_filter

        r = random()
        if gauss_blur_low <= r <= gauss_blur_high:
            im_n = im_n.filter(ImageFilter.GaussianBlur(1))
        elif blur_low < r <= blur_high:
            im_n = im_n.filter(ImageFilter.BLUR)
        elif smooth_low < r <= smooth_high:
            im_n = im_n.filter(ImageFilter.SMOOTH)
        elif smooth_more_low < r <= smooth_more_high:
            im_n = im_n.filter(ImageFilter.SMOOTH_MORE)
        elif rank_low < r <= rank_high:
            im_n = im_n.filter(ImageFilter.RankFilter(size=3, rank=7))
        else:
            pass
        return im_n


class NoiseLines(Noise):
    """Adds noise lines to the image i.e. lines randomly on the picture."""

    def __init__(self, color_range, n_lines_max=1, proba_line=0.33, name='NoiseLines'):
        """

        Arguments:
            color_range:

            n_lines_max:

            proba_line:

            name: A string,
                the name of the Layer.
        """

        if name is None:
            raise ValueError('')
        if color_range is None:
            raise ValueError
        if len(color_range.colors) == 0:
            raise ValueError
        if (not isinstance(n_lines_max, int)) or n_lines_max < 0:
            raise ValueError
        if not (isinstance(proba_line, float) or isinstance(proba_line, int)) or not 0 <= proba_line <= 1:
            raise ValueError

        super(NoiseLines, self).__init__()

        self.color_range = color_range
        self.n_lines_max = n_lines_max
        self.proba_line = proba_line

        self.name = name


    def call(self, img):

        def draw_line_dep(im, x1, y1, x2, y2, fill, width=1):
            draw = ImageDraw.Draw(im)
            draw.line((x1, y1, x2, y2), fill=fill, width=width)
            del draw
            return im

        if img is None: raise ValueError('img is None')

        n = randint(0, self.n_lines_max)
        for i in range(n):
            if random() > self.proba_line: continue
            x1 = randint(0, img.width)
            x2 = randint(0, img.width)
            y1 = randint(0, img.height)
            y2 = randint(0, img.height)
            width = randint(1, 10)
            fill = choice(self.color_range.colors)
            img = draw_line_dep(img, x1, y1, x2, y2, fill, width=width)

        return img


class Enhance(Noise):
    """Adds enhancements filters to the image."""

    def __init__(self, contrast=0, brightness=0, sharpness=0, color=0, name='Enhance'):
        """
        Arguments:
            contrast:

            brightness:

            sharpness:

            color:

            name: A string,
                the name of the Layer.
        """

        if name is None:
            raise ValueError()
        if not all([item is not None and (isinstance(item, float) or isinstance(item, int)) for item in [contrast, brightness, sharpness, color]]):
            raise ValueError
        if sum([contrast, brightness, sharpness, color]) > 1:
            raise ValueError
        if not all(0 <= item <= 1 for item in [contrast, brightness, sharpness, color]):
            raise ValueError
        super(Enhance, self).__init__()
        self.name = name

        self.contrast = contrast
        self.brightness = brightness
        self.sharpness = sharpness
        self.color = color

    def call(self, img):

        if img is None:
            raise ValueError('img is None')

        im_n = img.copy()

        r = random()
        contrast_low, contrast_high = 0, self.contrast
        brightness_low, brightness_high = contrast_high, contrast_high + self.brightness
        sharpness_low, sharpness_high = brightness_high, brightness_high + self.sharpness
        color_low, color_high = sharpness_high, sharpness_high + self.color

        if contrast_low <= r < contrast_high:
            factor_contrast = randint(5, 10)/10
            enhancer = ImageEnhance.Contrast(im_n)
            im_n = enhancer.enhance(factor_contrast)
        elif brightness_low <= r < brightness_high:
            factor_brightness = randint(5, 15)/10
            enhancer = ImageEnhance.Brightness(im_n)
            im_n = enhancer.enhance(factor_brightness)
        elif sharpness_low <= r < sharpness_high:
            factor_sharpen = randint(0, 20)/10
            enhancer = ImageEnhance.Sharpness(im_n)
            im_n = enhancer.enhance(factor_sharpen)
        elif color_low <= r < color_high:
            factor_color = randint(0, 20)/10
            enhancer = ImageEnhance.Color(im_n)
            im_n = enhancer.enhance(factor_color)
        else:
            pass

        return im_n
