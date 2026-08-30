from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'grasp_processor'

setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            [os.path.join('resource', package_name)]
        ),
        (
            os.path.join('share', package_name),
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'uois_model'),
            glob('uois_model/*.pth')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='TODO',
    maintainer_email='TODO@email.com',
    description='TODO: Package description',
    license='TODO: License declaration',

    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'grasp_processor = grasp_processor.grasp_processor:main',
        ],
    },
)
