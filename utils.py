import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm
import tensorflow as tf

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


class DataGenerator(tf.keras.utils.Sequence):
    """DataGenerator is input data going into model.fit and validation_data
    Every Sequence must implement the __getitem__ and the __len__ methods.
    The method __getitem__ should return a complete batch.
    If you want to modify your dataset between epochs you may implement on_epoch_end.
    """

    def __init__(self, image_ids, image_pixels, labels,
                 batch_size=1, shuffle=False, augment=False):
        self.image_ids = image_ids
        self.image_pixels = image_pixels
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.image_grid = self.form_image_grid()

        self.on_epoch_end()

    def __len__(self):
        """ is used to determine how many images there are in dataset.
        Python len() function returns the length of the object.
        This function internally calls __len__() function of the object.
        So we can use len() function with any object that defines __len__() function.
        """
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        """When the batch corresponding to a given index is called,
        the generator executes the __getitem__ method to generate it.
        i.e To get batch at position 'index'
        """

        # Generate indices of the batch
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of ids
        batch_ids = [self.image_ids[k] for k in indices]
        self.batch_ids = batch_ids

        # Generate data
        X, y = self.__data_generation(batch_ids)

        return X, y

    def on_epoch_end(self):
        """If you want to modify your dataset between epochs you may implement on_epoch_end"""

        self.indices = np.arange(len(self.image_ids))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_ids):
        """Produces batch-size of data """

        X, y = [], []

        # Generate data
        for image_id in batch_ids:
            pixels = self.image_pixels[image_id]
            bboxes = self.labels[image_id]

            if self.augment:
                pixels, bboxes = self.augment_image(pixels, bboxes)

            else:
                pixels = self.contrast_image(pixels)
                bboxes = self.form_label_grid(bboxes)

            X.append(pixels)
            y.append(bboxes)

        X = np.array(X)
        y = np.array(y)

        return X, y

    def form_image_grid(self):
        """creates image grid cells which indicate the information about the location where a cell is """

        image_grid = np.zeros((16, 16, 4))
        cell = [0, 0, 16, 16]

        for i in range(0, 16):
            for j in range(0, 16):
                image_grid[i, j] = cell

                cell[0] = cell[0] + cell[2]

            cell[0] = 0
            cell[1] = cell[1] + cell[3]

        return image_grid

    def augment_image(self, pixels, bboxes):
        """augments image

        Args:
            pixels: a batch size of images as array
            bboxes: a batch size of bboxes as array
        retruns:
            augmented images and bboxes scaled down 0 to 1,
        """

        # from 1024 to 256
        downsized_bboxes = bboxes / 4

        bbox_labels = np.ones(len(bboxes))
        aug_result = self.train_augmentations(image=pixels, bboxes=downsized_bboxes, labels=bbox_labels)
        bboxes = self.form_label_grid(aug_result['bboxes'])

        return aug_result['image'] / 256, bboxes

    def contrast_image(self, pixels):
        """converts images into grayscale"""

        aug_result = self.val_augmentations(image=pixels)
        return aug_result['image'] / 256

    def form_label_grid(self, bboxes):
        """returns Yolo shape of a label grid"""

        label_grid = np.zeros((16, 16, 5))

        for i in range(16):
            for j in range(16):
                cell = self.image_grid[i, j]
                label_grid[i, j] = self.rect_intersect(cell, bboxes)

        return label_grid

    def rect_intersect(self, cell, bboxes):
        """puts all boundary boxes into appropriate cells in the grid."""

        cell_x, cell_y, cell_width, cell_height = cell
        cell_x_max = cell_x + cell_width
        cell_y_max = cell_y + cell_height

        anchor_one = np.zeros(5)
        anchor_two = np.zeros(5)

        for bbox in bboxes:
            if self.augment:
                bbox_ = bbox
            else:
                bbox_ = bbox / 4
            box_x, box_y, box_width, box_height = bbox_
            box_x_centre = box_x + box_width / 2
            box_y_centre = box_y + box_height / 2

            # If the centre of box is in the cell,
            if cell_x <= box_x_centre < cell_x_max and cell_y <= box_y_centre < cell_y_max:

                if anchor_one[0] == 0:
                    anchor_one = self.yolo_shape(bbox_, cell)

                else:
                    break

        return anchor_one

    def yolo_shape(self, bbox, cell):
        """converts the shape of boundary boxes into the shape of Yolo """

        box_x, box_y, box_width, box_height = bbox
        cell_x, cell_y, cell_width, cell_height = cell

        box_x_centre = box_x + box_width / 2
        box_y_centre = box_y + box_height / 2

        resized_box_x = (box_x_centre - cell_x) / cell_width
        resized_box_y = (box_y_centre - cell_y) / cell_height
        resized_box_width = box_width / 256
        resized_box_height = box_height / 256

        return [1, resized_box_x, resized_box_y, resized_box_width, resized_box_height]


