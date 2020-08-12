import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm

DIR_INPUT = 'directory'


def group_boxes(group):
    """groups bbox by image_id and removes all without numeric data
    Args:
        group: Series of pandas
    Returns:
        Arrays grouped by the same image_id
    """
    boundaries = group['bbox'].str.split(',', expand=True)
    boundaries[0] = boundaries[0].str.slice(start=1)
    boundaries[3] = boundaries[3].str.slice(stop=-1)

    return boundaries.to_numpy().astype(float)


def load_image(image_id):
    """loads and resizes image to input size
    Args:
        image_id: An image id in train data
    return:
        resized image as array
    """
    input_size = (256, 256)
    image = Image.open(f'{DIR_INPUT}/train/{image_id}.jpg').resize(input_size)
    return np.asarray(image)


def reorganize(image_ids, labels):
    """separates image data to pixels and bboxes
    Args:
        image_ids: An iterator that contains ids of image
        labels: a dictionary that contains data of bboxes relevant to an image_ids
    return:
        resized image as array, bboxes
    """
    images = {}
    bboxes = {}
    for image_id in tqdm(image_ids):
        images[image_id] = load_image(image_id)
        bboxes[image_id] = labels[image_id]

    return images, bboxes


def draw_boxes_on_image(image, bboxes, color = 'red'):
    """draws lines on the picture where there are wheat
    Args:
        image: An image. Not array
        bboxes: an iterator of box data. (x,y,w,h)
        color : color of line
    return:
        image that lines are drawn on
    """
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]],
                       width=4, outline=color)
    return image


def prediction_to_submission(prediction, ids, threshold=0.2):
    """ The result of prediction doesn't have the same shape of submission.
    So It scales up to 1024,
    converts (centre_x, centre_y, width ,height) into (x, y, width, height),
    groups boundary boxes of images by a relevant id.

    Args:
        prediction: the result of prediction from the model
        ids: Ids of the test images
        threshold: to filter whether the prediction of a grid has bigger confidence than threshold
    returns:
        prediction modified as the form of submission
    """

    grid_x = prediction.shape[1]
    grid_y = prediction.shape[2]

    submission = {}

    for i, Id in enumerate(ids):
        converted_pred = []
        for j in range(grid_x):
            for k in range(grid_y):
                pred_ = prediction[i, j, k]
                if pred_[0] > threshold:
                    confidence = pred_[0]
                    cell_x = 64 * k
                    cell_y = 64 * j

                    box_width = pred_[3] * 1024
                    box_height = pred_[4] * 1024

                    box_x = cell_x + (pred_[1] * 64) - (box_width / 2)
                    box_y = cell_y + (pred_[2] * 64) - (box_height / 2)

                    converted_pred.append([confidence, box_x, box_y, box_width, box_height])

        submission[Id] = converted_pred

    return submission
