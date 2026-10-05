"""Offline geometry and execution guard tests (never moves an articulation)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import edge_tip_path as path
import run_lrbo_cube_edge as cube


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.model=path.URDFModel(Path(cube.__file__).parent.parent/'ur5e_robotiq_2f140.urdf')
        self.ratios={'robotiq_finger_joint':1, 'robotiq_left_inner_knuckle_joint':-1,
                     'robotiq_left_inner_finger_joint':1, 'robotiq_right_outer_knuckle_joint':-1,
                     'robotiq_right_inner_knuckle_joint':-1, 'robotiq_right_inner_finger_joint':1}

    def test_back_projection_and_base_transform(self):
        k=np.array([[200.,0,320],[0,200.,240],[0,0,1]])
        np.testing.assert_allclose(path.back_project((340,260),.8,k),[.08,.08,.8])
        transform=np.eye(4); transform[:3,:3]=path.rotation_axis_angle([0,0,1],np.pi/2)
        transform[:3,3]=[.1,.2,.3]
        np.testing.assert_allclose(path.transform_points(transform,[.08,.08,.8]),[.02,.28,1.1])
        for bad in (None,np.nan,np.inf,0,-1):
            with self.assertRaises(ValueError): path.back_project((340,260),bad,k)

    def test_actual_closed_tip_geometry(self):
        offset,axis,roll,points=self.model.closed_tip(self.ratios)
        np.testing.assert_allclose(offset,[0,0,.233160006],atol=1e-8)
        np.testing.assert_allclose(axis,[0,0,1],atol=1e-10)
        for local in points.values(): np.testing.assert_allclose(local,[0,.0325,0])
        self.assertAlmostEqual(np.dot(roll,axis),0)
        # Changing gripper closure changes offset: zero-q is not a fixed tool tip.
        tcp=self.model.pose('tool0',{})
        open_tip=np.mean([path.transform_points(np.linalg.inv(tcp)@self.model.pose(name,{}),p)
                          for name,p in points.items()],axis=0)
        self.assertGreater(np.linalg.norm(open_tip-offset),.01)

    def test_tilt_axis_mapping_for_non_z_tool_and_offset(self):
        for axis in (np.array([1.,0,0]),np.array([0,1.,0]),np.array([0,0,-1.])):
            roll=np.eye(3)[np.argmin(np.abs(axis))]
            r,edge,tool=path.tool_orientation([1,2,.2],[0,0,1],axis,roll,45,toward=[-2,1,0])
            np.testing.assert_allclose(r@axis,tool,atol=1e-9)
            self.assertAlmostEqual(np.dot(tool[:2],edge[:2]),0)
            self.assertAlmostEqual(tool[2],-np.sqrt(.5))
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(r),1)
            offset=np.array([.01,.02,.23]); desired=np.array([.3,.2,.15])
            tcp=desired-r@offset
            np.testing.assert_allclose(tcp+r@offset,desired,atol=1e-12)
            self.assertGreater(np.linalg.norm((r@offset)[:2]),.1)

    def test_quaternion_round_trip_including_pi(self):
        from run_lrbo_isaac import _rotation_error_deg
        # Use scene's same quaternion formula, without importing SimulationApp.
        def rotation(q):
            w,x,y,z=q
            return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                             [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                             [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        for angle in (0,.7,np.pi,np.pi-.000001):
            expected=path.rotation_axis_angle([1,2,3],angle)
            np.testing.assert_allclose(rotation(path.quaternion(expected)),expected,atol=1e-8)

    def test_plan_consistency_bounds_and_depth_rejection(self):
        k=np.array([[200.,0,100],[0,200.,100],[0,0,1]])
        edge={'p1':(80,100),'p2':(100,100)}
        depth={'p1_surface_m':.8,'p2_surface_m':.8,'valid_count':41,'sample_count':41,
               'samples':[{'edge_uv':(u,100),'depth_m':.8} for u in np.linspace(80,100,41)]}
        args=dict(edge=edge,depth_info=depth,intrinsics=k,base_camera=np.eye(4),world_base=np.eye(4),
                  d_tcp_tip=[0,0,.23316],tool_axis=np.array([0.,0,1]),roll_axis=np.array([1.,0,0]),
                  table_normal=[0,0,1],clearance=.01,tilt_deg=45,spacing=.002,toward=[0,1,0],
                  tip_bounds=([-1,-1,0],[1,1,2]),tcp_bounds=([-1,-1,0],[1,1,2]))
        plan=path.build_edge_plan(**args)
        np.testing.assert_allclose(plan['tip'],plan['tcp']+plan['rotation']@plan['offset'])
        np.testing.assert_allclose(plan['tip'][0],[-.08,0,.81])
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(plan['tip'],axis=0),axis=1)),.0020001)
        with self.assertRaisesRegex(ValueError,'workspace'):
            path.build_edge_plan(**(args|{'tip_bounds':([0,0,0],[1,1,1])}))
        depth['p1_surface_m']=np.nan
        with self.assertRaisesRegex(ValueError,'depth'): path.build_edge_plan(**args)
        depth['p1_surface_m']=.8; depth['samples'][20]['depth_m']=1.2
        with self.assertRaisesRegex(ValueError,'inconsistent'): path.build_edge_plan(**args)

    def test_lula_model_uses_exact_arm_and_fixed_mounts(self):
        with tempfile.TemporaryDirectory() as directory:
            descriptor,urdf=self.model.write_lula_arm_model(directory)
            reduced=path.URDFModel(urdf)
            joints=dict(zip(path.ARM_JOINTS,[.3,-1.2,1.1,-1.4,-1.1,.2]))
            np.testing.assert_allclose(reduced.pose('tool0',joints),self.model.pose('tool0',joints))
            self.assertIn('wrist_3-flange',urdf.read_text())
            self.assertIn('cspace:',descriptor.read_text())


class ExecutionGuardTests(unittest.TestCase):
    def test_vertical_turn_keeps_tip_fixed_for_both_edge_directions(self):
        for edge in (np.array([1.,0,0]),np.array([-1.,0,0])):
            rotation, _, _ = path.tool_orientation(edge,[0,0,1],[0,0,1],[1,0,0],45)
            offset=np.array([.003,-.002,.23316]); tip=np.array([.4,.1,.15])
            rotations,tcp,angle=path.fixed_tip_vertical_turn(rotation,offset,tip,edge,[0,0,1],[0,0,1])
            np.testing.assert_allclose(tcp+np.einsum('nij,j->ni',rotations,offset),
                                       np.tile(tip,(len(tcp),1)),atol=1e-12)
            np.testing.assert_allclose(rotations[-1]@[0,0,1],[0,0,-1],atol=1e-12)
            self.assertAlmostEqual(abs(angle),np.pi/4)

    def test_preview_flag_cannot_issue_robot_or_gripper_commands(self):
        executor=object.__new__(path.SlowEdgeExecutor)
        executor.settings=SimpleNamespace(EXECUTE_ROBOT=False)
        executor.status=Mock(); executor._apply=Mock(); executor._solver=Mock()
        executor.start({'anything':'preview'})
        executor._apply.assert_not_called(); executor._solver.assert_not_called()

    def test_ik_failure_before_arm_motion(self):
        # Fake solver fails before any command is applied; preflight must abort.
        executor=object.__new__(path.SlowEdgeExecutor)
        executor.settings=cube
        executor._solver=Mock(return_value=np.zeros(6))
        solver=Mock()
        solver.get_cspace_position_limits.return_value=(-np.ones(6)*6,np.ones(6)*6)
        solver.compute_inverse_kinematics.return_value=(np.zeros(6),False)
        executor.solver=solver
        current=np.eye(4); current[:3,3]=[.4,0,.5]
        executor._matrix=Mock(return_value=current); executor.tool_path='/tool0'
        executor.plan={'world_base':np.eye(4),'rotation':np.eye(3),'offset':np.array([0,0,.23316]),
                       'world_tip':np.array([[.4,0,.2],[.5,0,.2]]),
                       'world_tcp':np.array([[.4,0,.43316],[.5,0,.43316]]),
                       'turn_rotations':np.array([np.eye(3),np.eye(3)]),
                       'world_turn_tcp':np.array([[.5,0,.43316],[.5,0,.43316]]),
                       'tip_bounds':([-1,-1,0],[1,1,2]),'tcp_bounds':([-1,-1,0],[1,1,2])}
        executor._apply=Mock()
        with self.assertRaisesRegex(ValueError,'IK failed'): executor._preflight()
        executor._apply.assert_not_called()

    def test_new_detection_invalidates_old_plan(self):
        detector=cube.CubeEdgeDetector()
        detector.path_plan={'old':True}
        detector.request_detection()
        self.assertIsNone(detector.path_plan)
        self.assertTrue(detector._detect_requested)


if __name__=='__main__': unittest.main()
