import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# 判断是否是图片
def is_image_file(filename):
    # Return True if bool(x) is True for any x in the iterable
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 返回由文件(目录)名组成的列表和字典(键是目录名，值是序号)
def find_classes(dir):
    # os.listdir(path)用于返回path路径下的目录名组成的列表
    # os.path.isdir判断对象是否为一个目录
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# 返回值是一个列表，images的每一个元素都是一个元组，第一个值是图片路径，
# 第二个值是该图片所在的文件夹的序号
def make_dataset(dir, class_to_idx, class_num):
    images = []
    # Expand ~ and ~user constructs
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        # os.walk返回：当前路径(即d)，当前路径下的目录列表，当前路径下的文件列表
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    # 返回目录的序号
                    gxzeng_class = class_to_idx[target]
                    if gxzeng_class >= class_num:
                        break  # only read class_num
                    item = (path, gxzeng_class)
                    # images列表的每一个元素都是一个元组，第一个值是图片路径，
                    # 第二个值是该图片所在的文件夹的序号(即所属类别)
                    images.append(item)
    return images

# 把打开的图片转为RGB格式
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


# 制作数据集，返回值是：处理后的图片数据以及所属类别
class gx_ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None, class_num=0,
                 loader=default_loader):
        # 返回由文件(目录)名组成的列表和字典(键是目录名，值是序号)
        classes, class_to_idx = find_classes(root)
        # add by gxzeng
        classes = classes[0:class_num]
        # 返回值是一个列表，images的每一个元素都是一个元组，第一个值是图片路径，
        # 第二个值是该图片所在的文件夹(目录)的序号
        imgs = make_dataset(root, class_to_idx, class_num)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
