from setuptools import setup

package_name = 'ur5e_camera_workspace'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/ur5e_camera_workspace.yaml']),
        ('share/' + package_name + '/launch', ['launch/ur5e_camera_workspace.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/ur5e_camera_workspace.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': ['workspace_visualizer = ur5e_camera_workspace.workspace_visualizer:main']},
)