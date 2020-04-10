from .single_stage_rbbox_360 import SingleStageDetectorRbbox_360
from ..registry import DETECTORS


@DETECTORS.register_module
class RetinaNetRbbox_360(SingleStageDetectorRbbox_360):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetRbbox_360, self).__init__(backbone, neck, bbox_head, rbbox_head,
                                             train_cfg, test_cfg, pretrained)
