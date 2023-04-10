#!/bin/bash

# 引数の数をチェック
if [ $# -lt 2 ]; then
  echo "Error: At least 2 arguments is required." >&2
  exit 1
fi

RUN_MODE="$1"
model_version="$2"

MODEL_PATH="onnx/model_$model_version.onnx"
echo $MODEL_PATH


if [ "$RUN_MODE" = "all" ] || [ "$RUN_MODE" = "export" ]; then
    echo "1. モデルをonnx形式でエクスポート...."
    PYTHONPATH=. python ./src/tools/run_gphmer_handmesh_inference.py \
    --image_file_or_path ./samples/hand --device cpu \
    --resume_checkpoint ./models/graphormer_release/graphormer_hand_state_dict.bin \
    --export_model $MODEL_PATH
fi

if [ "$RUN_MODE" = "all" ] || [ "$RUN_MODE" = "infer" ]; then
    echo "2. onnx形式での推論のテストを実行...."
    # TEST_IMAGE_PATH_backup="../FastMETRO/demo/sample_hand_images_12/1.jpeg"

    # TEST_IMAGE_PATH="samples/hand2/02.jpeg"
    # TEST_IMAGE_PATH="samples/02/03.jpeg"
    # -------------------
    # TEST_IMAGE_PATH="samples/02/01.jpeg"
    # TEST_IMAGE_PATH="samples/02/05.jpeg"
    # -------------------
    # TEST_IMAGE_PATH="samples/03/04.png"
    # TEST_IMAGE_PATH="samples/03/05.png"
    # -------------------
    TEST_IMAGE_PATH="samples/03/06.png"

    # export PYTHONPATH=".:/Users/taichi.muraki/workspace/Python/mur6/FastMETRO"
    echo python scripts/infer_by_onnx_model.py --sample_dir $TEST_IMAGE_PATH --model_path $MODEL_PATH
    python scripts/infer_by_onnx_model.py --sample_dir $TEST_IMAGE_PATH --model_path $MODEL_PATH
fi
