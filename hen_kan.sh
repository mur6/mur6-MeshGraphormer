echo "1. モデルをonnx形式でエクスポート...."
# PYTHONPATH=. python ./src/tools/run_gphmer_handmesh_inference.py --image_file_or_path ./samples/hand --device cpu --resume_checkpoint ./models/graphormer_release/graphormer_hand_state_dict.bin --num 



echo "2. onnx形式での推論のテストを実行...."
TEST_IMAGE_PATH="../FastMETRO/demo/sample_hand_images_12/1.jpeg"
MODEL_PATH="onnx/model_$1.onnx"
echo $MODEL_PATH
PYTHONPATH=. python scripts/infer_by_onnx_model.py --sample_dir $TEST_IMAGE_PATH --model_path $MODEL_PATH
