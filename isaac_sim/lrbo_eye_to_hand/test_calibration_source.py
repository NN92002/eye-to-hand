import unittest
from unittest.mock import Mock, patch
import run_lrbo_cube_edge as cube

class SourceTests(unittest.TestCase):
    def detector(self):
        detector=object.__new__(cube.CubeEdgeDetector)
        detector.executor=None
        detector.calibration_source='OpenCV'
        detector._source_status=Mock()
        detector._invalidate_path=Mock()
        detector._set_path_status=Mock()
        return detector

    def test_switch_invalidates_old_plan(self):
        d=self.detector();d.select_calibration_source('LRBO')
        self.assertEqual(d.calibration_source,'LRBO')
        d._invalidate_path.assert_called_once()
        self.assertFalse(d._execute_requested)

    def test_execution_prevents_switch(self):
        d=self.detector();d.executor=Mock(active=True)
        d.select_calibration_source('LRBO')
        self.assertEqual(d.calibration_source,'OpenCV')
        d._invalidate_path.assert_not_called()

    def test_each_source_loads_its_own_file(self):
        d=self.detector();d._opencv_path_model=Mock();d._lrbo_path_model=Mock()
        d._opencv_path_model.get_value_as_string.return_value='opencv.yaml'
        d._lrbo_path_model.get_value_as_string.return_value='lrbo.json'
        with patch.object(cube,'load_calibration') as load:
            d.load_selected_calibration();load.assert_called_with('opencv.yaml')
            d.calibration_source='LRBO';d.load_selected_calibration();load.assert_called_with('lrbo.json')

if __name__=='__main__':unittest.main()
