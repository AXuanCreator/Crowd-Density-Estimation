import numpy as np
from torchvision import transforms

from models.can import CanAlexNet
from utils import (
    load_model,
    merge_density,
    save_density_image,
    save_image_with_contours,
)

image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

model = CanAlexNet().to("cuda")
load_model("./checkpoints/ckpt_500_160234.pth", model)


def generate_countours_rgb(image):
    h, w = image.shape[0:2]
    h_biset, w_biset = h // 2, w // 2
    img = image_transform(image.copy()).unsqueeze(0).to("cuda")
    img_parts = [
        img[:, :, :h_biset, :w_biset],
        img[:, :, h_biset:, :w_biset],
        img[:, :, :h_biset, w_biset:],
        img[:, :, h_biset:, w_biset:],
    ]
    output_list = [model(img_part) for img_part in img_parts]
    merge = merge_density(
        [density.squeeze().detach().cpu().numpy() for density in output_list]
    )
    binary_x = save_density_image(x=merge, binarization=True, up_sample=8)
    contours_rgb = save_image_with_contours(binary_x=binary_x, rgb=image)

    return contours_rgb


if __name__ == "__main__":
    from PIL import Image

    img = Image.open("./debug/test.png").convert("RGB")
    img = np.array(img)
    x = generate_countours_rgb(image=img)
    print(x)
