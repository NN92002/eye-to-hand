import unittest
from pathlib import Path
import numpy as np
from edge_tip_path import URDFModel, ARM_JOINTS, PAD_FRAMES, transform_points, rotation_axis_angle

class CollisionGeometryTests(unittest.TestCase):
    def test_reported_forearm_collision_is_not_present_in_urdf_geometry(self):
        model=URDFModel(Path(__file__).parent.parent/'ur5e_robotiq_2f140.urdf')
        geometry=dict(model.collision_vertices())
        self.assertTrue(set(PAD_FRAMES).issubset(geometry))
        self.assertEqual(len(geometry),18)
        q=[-.8418993163611876,-2.1787725678716665,2.5914043094976407,
           -2.620209988132613,-1.07358053318266,.9988838364275043]
        base=np.eye(4)
        base[:3,:3]=rotation_axis_angle([0,0,1],np.pi/4)
        base[:3,3]=[.1,-.35,.03]
        points=transform_points(base@model.pose('forearm_link',dict(zip(ARM_JOINTS,q))),geometry['forearm_link'])
        self.assertGreater(points[:,2].min(),.34)
        self.assertTrue(np.isfinite(points).all())

if __name__=='__main__': unittest.main()
