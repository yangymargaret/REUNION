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
					  "scikit-learn==1.1.3",
					  "numpy",
					  "scipy",
					  "pandas",
					  "pyranges",
					  "phenograph",
					  "pingouin",
					  "shap",
					  "ipywidgets",
					],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.8.0'
)




