# 美甲生成服务 API 文档

## 概述

美甲生成服务提供基于AI的美甲颜色生成功能，支持异步任务处理和实时进度查询。**新增边缘引导和ControlNet结构控制功能，显著提升掩码质量和AI渲染真实感。**

**基础URL**: `http://localhost:80`

## 通用响应格式

所有API接口都使用统一的响应格式：

```json
{
  "statusCode": 200,        // 状态码：200表示成功，-1表示失败
  "message": "操作成功",     // 响应消息
  "data": {},              // 响应数据
  // 其他字段根据接口而定
}
```

## 图像数据格式

**重要更新**: 所有图像数据现在使用完整的Data URL格式，而不是纯base64编码。

### Data URL格式
```
data:image/png;base64,{base64编码的图像数据}
```

### 前端使用示例
```javascript
// 直接使用，无需额外处理
const img = new Image();
img.src = response.data;  // response.data 已经是完整的Data URL
document.body.appendChild(img);
```

## API 接口列表

### 1. 提交美甲生成任务

**接口地址**: `POST /edit_nail`

**功能描述**: 提交美甲生成任务，返回任务ID用于后续查询

**请求参数**:
- `Content-Type`: `application/x-www-form-urlencoded`
- `img`: 原图像base64编码字符串
- `ref_img`: 参考色图像base64编码字符串

**请求示例**:
```bash
curl -X POST http://localhost:80/edit_nail \
  -F "img=data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..." \
  -F "ref_img=data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
```

**响应格式**:

成功响应:
```json
{
  "statusCode": 200,
  "message": "任务已提交",
  "task_id": "1430523456789",
  "data": ""
}
```

失败响应:
```json
{
  "statusCode": -1,
  "message": "原图像或参考色图像未提供",
  "data": ""
}
```

### 2. 查询任务进度

**接口地址**: `GET /get_progress/{task_id}`

**功能描述**: 查询指定任务的处理进度

**路径参数**:
- `task_id`: 任务ID（从提交任务接口返回）

**请求示例**:
```bash
curl http://localhost:80/get_progress/1430523456789
```

**响应格式**:

任务处理中:
```json
{
  "statusCode": 200,
  "task_id": "1430523456789",
  "progress": 0.65,
  "current_step": 13,
  "total_steps": 20,
  "message": "AI生成进度: 65.0% (13/20)",
  "is_completed": false
}
```

任务已完成:
```json
{
  "statusCode": 200,
  "task_id": "1430523456789",
  "progress": 1.0,
  "current_step": 0,
  "total_steps": 0,
  "message": "生成完成",
  "is_completed": true,
  "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."  // 完整的Data URL格式
}
```

任务不存在:
```json
{
  "statusCode": -1,
  "message": "任务不存在",
  "task_id": "1430523456789"
}
```

### 3. 获取任务结果

**接口地址**: `GET /get_result/{task_id}`

**功能描述**: 获取已完成任务的结果图像

**路径参数**:
- `task_id`: 任务ID

**请求示例**:
```bash
curl http://localhost:80/get_result/1430523456789
```

**响应格式**:

成功响应:
```json
{
  "statusCode": 200,
  "task_id": "1430523456789",
  "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",  // 完整的Data URL格式
  "message": "生成完成"
}
```

失败响应:
```json
{
  "statusCode": -1,
  "message": "任务不存在或未完成",
  "task_id": "1430523456789"
}
```

### 4. 清理任务

**接口地址**: `GET /cleanup_task/{task_id}`

**功能描述**: 清理指定任务的相关文件和内存数据

**路径参数**:
- `task_id`: 任务ID

**请求示例**:
```bash
curl http://localhost:80/cleanup_task/1430523456789
```

**响应格式**:

成功响应:
```json
{
  "statusCode": 200,
  "message": "任务清理完成",
  "task_id": "1430523456789"
}
```

失败响应:
```json
{
  "statusCode": -1,
  "message": "清理失败的具体原因"
}
```

## 使用流程

1. **提交任务**: 调用 `POST /edit_nail` 提交原图像和参考色图像，获取任务ID
2. **查询进度**: 使用任务ID调用 `GET /get_progress/{task_id}` 查询处理进度
3. **获取结果**: 任务完成后调用 `GET /get_result/{task_id}` 获取结果图像
4. **清理资源**: 使用完毕后调用 `GET /cleanup_task/{task_id}` 清理相关文件

## 前端集成示例

### JavaScript示例
```javascript
// 1. 提交任务
async function submitTask(originalImage, referenceImage) {
    const formData = new FormData();
    formData.append('img', originalImage);
    formData.append('ref_img', referenceImage);
    
    const response = await fetch('/edit_nail', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result.task_id;
}

// 2. 查询进度
async function checkProgress(taskId) {
    const response = await fetch(`/get_progress/${taskId}`);
    const result = await response.json();
    
    if (result.is_completed && result.data) {
        // 直接使用Data URL，无需额外处理
        const img = new Image();
        img.src = result.data;  // 已经是完整的Data URL
        document.body.appendChild(img);
        return true;
    }
    
    return false;
}

// 3. 轮询进度
async function pollProgress(taskId) {
    const interval = setInterval(async () => {
        const isCompleted = await checkProgress(taskId);
        if (isCompleted) {
            clearInterval(interval);
            console.log('任务完成！');
        }
    }, 1000);
}
```

### Python示例
```python
import requests
import base64

# 1. 提交任务
def submit_task(original_image_path, reference_image_path):
    with open(original_image_path, 'rb') as f:
        original_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    with open(reference_image_path, 'rb') as f:
        reference_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    response = requests.post('http://localhost:80/edit_nail', data={
        'img': f'data:image/jpeg;base64,{original_b64}',
        'ref_img': f'data:image/jpeg;base64,{reference_b64}'
    })
    
    return response.json()['task_id']

# 2. 获取结果
def get_result(task_id):
    response = requests.get(f'http://localhost:80/get_result/{task_id}')
    result = response.json()
    
    if result['statusCode'] == 200:
        # 从Data URL中提取base64数据
        data_url = result['data']
        b64_data = data_url.split(',')[1]
        
        # 解码并保存图像
        import base64
        image_data = base64.b64decode(b64_data)
        with open('result.png', 'wb') as f:
            f.write(image_data)
        
        return 'result.png'
    
    return None
```

## 新增功能说明

### 边缘引导掩码增强
- **功能**: 使用Canny/Sobel边缘检测辅助掩码边界优化
- **效果**: 提升掩码与真实指甲的贴合度
- **自动应用**: 在掩码生成流程中自动启用

### ControlNet结构控制
- **功能**: 生成多种结构图约束AI渲染效果
- **效果**: 增强AI渲染的真实感和3D效果
- **自动应用**: 在AI渲染流程中自动启用

## 错误码说明

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| -1 | 请求失败 |

## 注意事项

1. **图像数据格式**: 所有图像数据现在使用完整的Data URL格式
2. **前端兼容性**: 前端可以直接使用返回的Data URL，无需额外处理
3. **任务ID**: 任务ID是唯一的，用于标识整个处理流程
4. **自动增强**: 边缘引导和ControlNet结构控制功能会自动应用，无需额外配置
5. **调试输出**: 增强功能会生成调试图像，保存在相应目录中

## 更新日志

### v2.0.0 (最新)
- ✅ 图像数据格式改为完整的Data URL格式
- ✅ 新增边缘引导掩码增强功能
- ✅ 新增ControlNet结构控制功能
- ✅ 优化掩码生成质量
- ✅ 提升AI渲染真实感
- ✅ 简化前端集成流程