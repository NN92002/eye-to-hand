import unittest
import numpy as np
from edge_tip_path import transform_points,tool_orientation,rotation_axis_angle
from edge_path_evaluation import evaluate_path,project,cube_edges


class EvaluationTests(unittest.TestCase):
    def case(self,shift=None):
        cube=np.eye(4); cube[:3,:3]*=.12; cube[:3,3]=[.45,0,.09]
        gt=np.eye(4); gt[:3,:3]=np.diag([1,-1,-1]); gt[:3,3]=[.45,-.3,1]
        k=np.array([[300.,0,320],[0,300.,240],[0,0,1]])
        ideal=np.array([[.414,-.06,.15],[.486,-.06,.15]])  # 60% of top-front edge
        uv,_=project(ideal,np.linalg.inv(gt),k)
        actual=ideal.copy()
        if shift is not None: actual+=shift
        r,e,d=tool_orientation(actual[1]-actual[0],[0,0,1],np.array([0.,0,1]),np.array([1.,0,0]),45,[0,1,0])
        points=actual[0]+np.linspace(0,1,41)[:,None]*(actual[1]-actual[0])
        tip=points+[0,0,.01]; offset=np.array([0,0,.23316])
        plan={'world_base':np.eye(4),'pixels':uv,'base':actual,'tip':tip,
              'tcp':tip-r@offset,'rotation':r,'edge_direction':e,'offset':offset,
              'calibration_source':'OpenCV','calibration_file':'example.yaml'}
        return plan,cube,gt,k

    def evaluate(self,shift=None):
        plan,cube,gt,k=self.case(shift)
        return evaluate_path(plan,cube,1.,gt,k,np.array([0.,0,1]),np.array([1.,0,0]),[0,0,1],.01,45)

    def test_zero_error_and_partial_segment_not_full_edge(self):
        report=self.evaluate()
        self.assertTrue(report['matched'],report)
        self.assertAlmostEqual(report['coverage_percent'],60,places=5)
        self.assertAlmostEqual(report['ideal_full_edge_length_mm'],120,places=5)
        self.assertAlmostEqual(report['ideal_subsegment_length_mm'],72,places=5)
        self.assertLess(report['path_rms_error_mm'],1e-8)
        self.assertLess(report['tcp_rms_error_mm'],1e-8)

    def test_report_accepts_each_selected_source_without_legacy_opencv_key(self):
        for source,filename in [('OpenCV','example.yaml'),('LRBO','example.json')]:
            plan,cube,gt,k=self.case()
            plan['calibration_source']=source;plan['calibration_file']=filename
            report=evaluate_path(plan,cube,1.,gt,k,np.array([0.,0,1]),np.array([1.,0,0]),[0,0,1],.01,45)
            self.assertTrue(report['matched'])
            self.assertEqual(report['calibration_source'],source)
            self.assertEqual(report['calibration_file'],filename)

    def test_known_translation_error_is_not_corrected(self):
        report=self.evaluate([0,.003,.004])
        self.assertAlmostEqual(report['path_rms_error_mm'],5,places=5)
        self.assertAlmostEqual(report['tcp_rms_error_mm'],5,places=5)
        np.testing.assert_allclose(report['endpoint_delta_est_minus_ideal_mm'],[[0,3,4],[0,3,4]],atol=1e-8)

    def test_false_rgb_line_rejected(self):
        plan,cube,gt,k=self.case()
        plan['pixels']=np.asarray(plan['pixels'])+[0,35]
        report=evaluate_path(plan,cube,1.,gt,k,np.array([0.,0,1]),np.array([1.,0,0]),[0,0,1],.01,45)
        self.assertFalse(report['matched'],report)

    def test_cube_scale_applied_once_and_rotated(self):
        _,cube,gt,k=self.case()
        cube[:3,:3]=rotation_axis_angle([0,0,1],.7)*.12
        edges=cube_edges(cube,np.eye(4),gt[:3,3],1.)
        self.assertEqual(len(edges),12)
        for edge in edges:
            self.assertAlmostEqual(np.linalg.norm(edge['base'][1]-edge['base'][0]),.12)
        self.assertTrue(any(not e['visible'] for e in edges))

    def test_reversed_endpoint_order(self):
        plan,cube,gt,k=self.case()
        plan['pixels']=plan['pixels'][::-1]; plan['base']=plan['base'][::-1]
        plan['tip']=plan['tip'][::-1]; plan['edge_direction']*=-1
        r,e,d=tool_orientation(plan['base'][1]-plan['base'][0],[0,0,1],np.array([0.,0,1]),np.array([1.,0,0]),45,[0,1,0])
        plan['rotation']=r; plan['tcp']=plan['tip']-r@plan['offset']
        report=evaluate_path(plan,cube,1.,gt,k,np.array([0.,0,1]),np.array([1.,0,0]),[0,0,1],.01,45)
        self.assertTrue(report['matched'])
        self.assertLess(report['path_rms_error_mm'],1e-8)


if __name__=='__main__': unittest.main()
