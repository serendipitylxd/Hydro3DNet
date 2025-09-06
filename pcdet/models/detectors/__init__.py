from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .pv_rcnn_plusplus_2 import PVRCNNPlusPlus2
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet
from .voxelnext import VoxelNeXt
from .transfusion import TransFusion
from .bevfusion import BevFusion
from .CT3D import CT3D
from .CT3D_3CAT import CT3D_3CAT
from .pg_rcnn import PGRCNN
from .fusion import FUSION
from .voxel_rcnn_centerhead import VoxelRCNNCENTERHEAD
from .voxel_trout import VOXELTROUT
from .hydro_3d_net import Hydro3DNet

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'PVRCNNPlusPlus2': PVRCNNPlusPlus2,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'PillarNet': PillarNet,
    'VoxelNeXt': VoxelNeXt,
    'TransFusion': TransFusion,
    'BevFusion': BevFusion,
    'CT3D': CT3D,
    'CT3D_3CAT': CT3D_3CAT,
    'PGRCNN': PGRCNN,
    'FUSION': FUSION,
    'VoxelRCNNCENTERHEAD': VoxelRCNNCENTERHEAD,
    'VOXELTROUT': VOXELTROUT,
    'Hydro3DNet': Hydro3DNet,


}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
