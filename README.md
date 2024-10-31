# Report and code by Thomas Selin for project 1 in the university course `Multivariate data analysis in engineering` at `Mälardalens University`

An isolation forest machine learning model for anomaly detection in an IT service component (microservice) was created.

See the [Project Report](Project_report.pdf) for a detailed report on the execution, result and conclusions of the project.

## Notes:

- PCA was performed, but as it didn't help much in dimensionality reduction and would lead to a less explainable model, that result was not used when creating the final model. For completeness, the code is still kept.

- The dataset is not included as consists of somewhatwhat sensitive company internal data. See the [Project Report](Project_report.pdf) for some further insight into the data.

## Improved version
An improved version of the code, with focus on code quality, modularity etc., was added after project completion in the `improved_model_creation.py` file. I split the data and did training and testing separately. In the `improved_model_creation_outputs` you also can get insight into the data.
