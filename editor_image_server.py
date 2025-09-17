import os
import sys
import cv2
import numpy as np
import base64
import json
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV
from nail_color_transfer import U2NetMasker
from color_nail_full_pipeline_adapter import run_full_pipeline

app = Flask(__name__)

# 配置最大请求大小限制 (100MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
# 配置表单内存大小限制 (50MB)
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * 1024 * 1024
# 配置最大表单部分数量
app.config['MAX_FORM_PARTS'] = 1000

os.makedirs("data/test_images", exist_ok=True)
os.makedirs("data/test_masks", exist_ok=True)
os.makedirs("data/reference", exist_ok=True)
os.makedirs("data/output/final", exist_ok=True)

# 初始化处理器
nail = NailSDXLInpaintOpenCV()
masker = U2NetMasker()

def process_with_progress(task_id, img_path, ref_path, mask_path):
    """带进度回调的处理函数"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始处理任务{task_id}: img={img_path}, ref={ref_path}, mask={mask_path}")
    
    # 初始化进度（只设置状态和消息，不设置progress字段）
    task_progress[task_id] = {
        "status": "processing",
        "progress": 0.0,  # 仅用于AI阶段
        "current_step": 0,
        "total_steps": 0,
        "message": "开始处理..."
    }
    
    try:
        def progress_callback(progress, current_step, total_steps):
            # 只在AI精炼阶段更新进度
            task_progress[task_id].update({
                "progress": progress,
                "current_step": current_step,
                "total_steps": total_steps,
                "message": f"AI精炼进度: {progress:.1%} ({current_step}/{total_steps})"
            })
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 任务{task_id}进度更新: AI精炼进度: {progress:.1%} ({current_step}/{total_steps})")
        # 检查掩码是否存在，如果不存在则用 U2Net 生成
        if not os.path.exists(mask_path):
            img = cv2.imread(str(img_path))
            if img is None:
                raise Exception(f"无法读取图片: {img_path}")
            mask = masker.get_mask(img, str(img_path), disable_cache=True)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(str(mask_path), mask)
        # 调用新主流程，传递任务ID确保文件唯一性
        run_full_pipeline(img_path, ref_path, mask_path, progress_callback, task_id)
        # 读取生成结果
        stem = Path(img_path).stem
        final_path = os.path.join("data/output/final", f"{stem}_final.png")
        if os.path.exists(final_path):
            with open(final_path, "rb") as f:
                final_b64 = base64.b64encode(f.read()).decode("utf-8")
            final_data_url = f"data:image/png;base64,{final_b64}"
            task_results[task_id] = {
                "status": "completed",
                "result": final_data_url,
                "message": "生成完成"
            }
        else:
            task_results[task_id] = {
                "status": "failed",
                "result": "",
                "message": "生成最终图像失败"
            }
    except Exception as e:
        task_results[task_id] = {
            "status": "failed",
            "result": "",
            "message": str(e)
        }
    if task_id in task_progress:
        del task_progress[task_id]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 任务{task_id}处理线程结束")

@app.route("/edit_nail", methods=["POST"])
def edit_nail():
    """
    美甲主流程生成接口（同步阻塞，直到AI精炼完成返回结果）
    
    **请求方式**：POST
    **URL**：/edit_nail
    **Content-Type**：application/x-www-form-urlencoded 或 multipart/form-data

    **请求参数**：
      - img:  原始手部图片，base64编码（不带 data:image/png;base64, 前缀）
      - ref_img:  参考色块图片，base64编码（不带 data:image/png;base64, 前缀）

    **返回值**（JSON）：
      - statusCode: 200 表示成功，-1 表示失败
      - message:    结果说明
      - task_id:    本次任务ID（可用于日志追踪）
      - data:       生成的最终美甲效果图，base64编码的data url（如 data:image/png;base64,...）

    **成功返回示例**：
    {
        "statusCode": 200,
        "message": "生成完成",
        "task_id": "143022123",
        "data": "data:image/png;base64,iVBORw0KGgoAAAANS..."
    }

    **失败返回示例**：
    {
        "statusCode": -1,
        "message": "图像解码失败",
        "task_id": "143022123",
        "data": ""
    }

    **说明**：
    - 本接口为同步阻塞，调用后需等待完整流水线（像素迁移+高光+AI精炼）全部完成。
    - 推荐前端用 FormData 或 x-www-form-urlencoded 方式提交。
    - 返回的 data 字段可直接用于 <img src="..."> 显示。
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] POST /edit_nail - 收到美甲生成请求")
    try:
        img_b64 = request.form.get("img")
        ref_b64 = request.form.get("ref_img")
        if not img_b64 or not ref_b64:
            return jsonify({"statusCode": -1, "message": "原图像或参考色图像未提供", "data": ""})
        img_data = base64.b64decode(img_b64)
        ref_data = base64.b64decode(ref_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        ref_img = cv2.imdecode(np.frombuffer(ref_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or ref_img is None:
            return jsonify({"statusCode": -1, "message": "图像解码失败", "data": ""})
        task_id = time.strftime("%H%M%S") + str(int(time.time() * 1000) % 1000).zfill(3)
        img_path = os.path.join("data/test_images", task_id + ".jpg")
        mask_path = os.path.join("data/test_masks", task_id + "_mask_input_mask.png")
        ref_path = os.path.join("data/reference", task_id + "_reference.jpg")
        cv2.imwrite(img_path, img)
        cv2.imwrite(ref_path, ref_img)
        # 每次请求都重新生成掩码
        mask = masker.get_mask(img, str(img_path), disable_cache=True)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(str(mask_path), mask)
        # 同步调用主流程，等待AI精炼完成
        run_full_pipeline(img_path, ref_path, mask_path)
        stem = Path(img_path).stem
        final_path = os.path.join("data/output/final", f"{stem}_final.png")
        if os.path.exists(final_path):
            with open(final_path, "rb") as f:
                final_b64 = base64.b64encode(f.read()).decode("utf-8")
            final_data_url = f"data:image/png;base64,{final_b64}"
            return jsonify({
                "statusCode": 200,
                "message": "生成完成",
                "task_id": task_id,
                "data": final_data_url
            })
        else:
            return jsonify({
                "statusCode": -1,
                "message": "生成最终图像失败",
                "task_id": task_id,
                "data": ""
            })
    except Exception as e:
        return jsonify({"statusCode": -1, "message": str(e), "data": ""})

if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 美甲生成服务器启动中...")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 服务器地址: http://0.0.0.0:80")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 可用API端点:")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   - POST /edit_nail - 提交美甲生成任务")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 服务器启动完成，等待请求...")
    app.run(host="0.0.0.0", port=80, debug=False) 