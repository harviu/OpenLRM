# 0 for multiple camera, 1 for one camera
F:\colmap\COLMAP.bat feature_extractor --database_path ./database.db --image_path . --ImageReader.single_camera 1
F:\colmap\COLMAP.bat exhaustive_matcher --database_path ./database.db
F:\colmap\COLMAP.bat mapper --database_path ./database.db --image_path ./images --output_path ./sparse
F:\colmap\COLMAP.bat model_converter --input_path ./sparse/0 --output_path ./sparse/text --output_type TXT