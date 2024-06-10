import setuptools

setuptools.setup(
	name='REUNION',
	version="0.1.0",
	author="Maggie",
	author_email="yangy4@mskcc.org",
	description="Python package for regulatory association inference",
	url="https://github.com/yangymargaret/REUNION",
	packages=['REUNION'],
	install_requires=["scanpy",
					  "anndata",
					  "scikit-learn",
					  "numpy",
					  "scipy",
					  "pandas",
					  "pyranges",
					  "phenograph",
					  "pingouin",
					],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.7.0',
	zip_safe=False
)


