import torchvision.transforms as transforms


normalize_3c = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
normalize_1c = transforms.Normalize(mean=[0.5],
                                     std=[0.5])

def transform_train_2d():

    return transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                                     std=[0.5]),
    ])

def transform_test_2d():
    return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                     std=[0.5]),
        ])