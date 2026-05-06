
def _mammo_transform_params() -> dict[str, list[str] | dict[str, list[float]]]:
    conf_params = {
        "train_transforms": ["load","channelFirst","ApplyVoi","make8Bit","resize","randGaus","Make3Channel","norm","rotate","flip"],
        "test_transforms": ["load", "channelFirst","ApplyVoi","make8Bit","resize","Make3Channel"],
        "transform_conf": {
            "use_mask": 0,
            "norm_mu": [0.485, 0.456, 0.406],
            "norm_std": [0.229, 0.224, 0.225],
            "brightness": [0.8, 1.2],
            "saturation": [0.8, 1.1],
            "contrast": [0.8, 1.2],
            "img_shape": [512,512],
        },
    }
    return conf_params