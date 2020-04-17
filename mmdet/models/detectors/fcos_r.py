from ..registry import DETECTORS
from .single_stage_rbbox_360 import SingleStageDetectorRbbox_360


@DETECTORS.register_module
class FCOS_R(SingleStageDetectorRbbox_360):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOS_R, self).__init__(backbone, neck, bbox_head, rbbox_head, train_cfg,
                                   test_cfg, pretrained)
