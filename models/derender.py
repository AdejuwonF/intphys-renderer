
# from structure.attributes import build_attributes
from torch import nn
from torchvision.models import resnet34


class Derender(nn.Module):
    """
    Neural Network used for obtaining object-centric representations from scenes
    """

    def __init__(self, cfg, attributes):
        """
        Initializes basic network topology (based on Resnet 34)

        Args:
            cfg: YACS configuration with experiment parameters
        """

        super(Derender, self).__init__()

        resnet = resnet34(pretrained=True)
        resnet_layers = list(resnet.children())

        # Replace first and last layer of Resnet
        resnet_layers.pop()
        resnet_layers.pop(0)
        resnet_layers.insert(0, nn.Conv2d(cfg.MODEL.IN_CHANNELS, 64, kernel_size=3, stride=2, padding=1))
        resnet_layers[-1] = nn.AvgPool2d(kernel_size=cfg.MODEL.POOLING_KERNEL_SIZE)
        # print(resnet_layers)

        self.backbone = nn.Sequential(*resnet_layers)

        self.add_camera = hasattr(attributes, "camera_terms")
        num_feats = cfg.MODEL.FEATURE_CHANNELS + 6 if self.add_camera else cfg.MODEL.FEATURE_CHANNELS

        middle_layers = [nn.Linear(num_feats, cfg.MODEL.MID_CHANNELS),
                         nn.ReLU()]
        if cfg.MODEL.NUM_MID_LAYERS > 1:
            middle_layers += [nn.Linear(cfg.MODEL.MID_CHANNELS,cfg.MODEL.MID_CHANNELS),
                              nn.ReLU()] * (cfg.MODEL.NUM_MID_LAYERS - 1)

        self.middle_layers = nn.Sequential(*middle_layers)
        self.final_layer = nn.Linear(cfg.MODEL.MID_CHANNELS, len(attributes))

        self.loss = attributes.loss
        self.attributes = attributes

    def forward(self, inputs):
        """
        Derenderer's forward pass

        Args:
            inputs: Inputs to compute forward pass on containing image maps and continuous as well as categorical
                    attributes

        Returns:

        """
        img_tuple = inputs["img_tuple"]
        x = self.backbone(img_tuple)
        x = x.view(x.size(0), -1)
        if self.add_camera:
            x = self.attributes.add_camera(x,inputs)
        x = self.middle_layers(x)
        x = self.final_layer(x)

        output = self.attributes.backward(x)


        if self.training:
            loss_dict = self.attributes.loss(output, inputs["attributes"])
        else:
            loss_dict = 0

        return {"output": output, "loss_dict": loss_dict}
